import sys
import torch
import yaml
import numpy as np
import pandas as pd
import joblib
import holidays
import traceback
from flask import Flask, jsonify, request
from influxdb_client import InfluxDBClient
from datetime import timedelta
import argparse
from waitress import serve
from urllib.parse import urlparse
import logging
import os

from model import NodePredictorNN
from config import MODEL_CONFIG

MODE = 'test'  # 可选值: 'service' 或 'test'

# def load_config():
#     parser = argparse.ArgumentParser(
#         description='Node predictor service',
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
    
#     parser.add_argument(
#         '-c', '--config',
#         default='config.yaml',
#         help='Path to configuration file'
#     )

#     args = parser.parse_args()
    
#     try:
#         with open(args.config, "r") as f:
#             config = yaml.safe_load(f)
            
#         logging.basicConfig(
#             filename=config["Predictor"]["LogFile"],
#             level=logging.INFO,
#             format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#         )
        
#         logging.info(f"Loading config from {args.config}...")
#         return config
#     except Exception as e:
#         raise RuntimeError(f"Failed to load config from {args.config}: {str(e)}")


# CONFIG = load_config()

# logging.info(f"CONFIG: {CONFIG}")

# DEBUG = CONFIG["Predictor"]["Debug"]
# url = urlparse(CONFIG["Predictor"]["URL"])
# PORT = url.port or 5000

# CHECKPOINT_PATH = CONFIG["Predictor"]["CheckpointPath"]
# SCALERS_PATH = CONFIG["Predictor"]["ScalersPath"]

# INFLUX_CONFIG = {
#     "url": CONFIG["InfluxDB"]["URL"],
#     "token": CONFIG["InfluxDB"]["Token"],
#     "org": CONFIG["InfluxDB"]["Org"],
#     "bucket": CONFIG["InfluxDB"]["Bucket"],
# }

# # 这里的cpu个数很突兀
PREDICTION_CONFIG = {
    "cpus_per_node": 32,
    "forecast_minutes": 30,
    "lookback_minutes": 240,
}


class NodePredictor:
    def __init__(self):
        # 根据运行模式选择配置
        if MODE == 'test':
            self.config = MODEL_CONFIG
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.feature_names = [
                'running_job_count',
                'active_node_count',
                'avg_req_cpu_rate',
                'avg_req_node_per_job',
                'avg_req_cpu_per_job',
                'avg_runtime_minutes',
            ]
            self.feature_size = len(self.feature_names)
            self._init_test_model()
        else:
            # 原有的服务模式初始化
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.feature_names = [
                "running_job_count",
                "active_node_count",
                "avg_req_cpu_rate",
                "avg_req_cpu_per_job",
                "avg_req_node_per_job",
                "avg_runtime_minutes",
            ]
            self.feature_size = len(self.feature_names)
            self.influx_client = InfluxDBClient(
                url=INFLUX_CONFIG["url"],
                token=INFLUX_CONFIG["token"],
                org=INFLUX_CONFIG["org"],
            )
            self._init_model()

    def _init_test_model(self):
        """初始化测试模式的模型"""
        try:
            # 创建模型实例
            self.model = NodePredictorNN(feature_size=self.feature_size).to(self.device)
            
            # 加载模型权重
            model_path = os.path.join(self.config['model_dir'], 'checkpoint.pth')
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
                
            self.model.eval()
            
            # 加载数据缩放器
            data_filename = os.path.splitext(os.path.basename(self.config['data_path']))[0]
            scaler_path = os.path.join(
                os.path.dirname(self.config['data_path']), 
                data_filename, 
                'dataset_scalers.pkl'
            )
            print(f"加载数据缩放器: {scaler_path}")
            scalers = joblib.load(scaler_path)
            self.feature_scaler = scalers['feature_scaler']
            self.dayback_scaler = scalers['dayback_scaler']
            self.target_scaler = scalers['target_scaler']
            
            print("测试模式：模型和数据缩放器加载成功")
            
        except Exception as e:
            logging.error("测试模式：模型加载失败")
            logging.error(f"错误类型: {type(e).__name__}")
            logging.error(f"错误信息: {str(e)}")
            logging.error(traceback.format_exc())
            raise RuntimeError(f"Failed to load test model: {str(e)}")

    def _init_model(self):
        self.model = NodePredictorNN(feature_size=self.feature_size).to(self.device)

        try:
            checkpoint = torch.load(
                CHECKPOINT_PATH, map_location=self.device, weights_only=True
            )

            if "model_state_dict" in checkpoint:
                logging.info("Loading model state dict from checkpoint")
                model_state = checkpoint["model_state_dict"]
                self.model.load_state_dict(model_state, strict=False)
            else:
                logging.info("Loading model from checkpoint")
                self.model.load_state_dict(checkpoint, strict=False)

            self.model.eval()

            scalers = joblib.load(SCALERS_PATH)
            logging.info("Loading scalers from path: %s", SCALERS_PATH)
            self.feature_scaler = scalers["feature_scaler"]
            self.dayback_scaler = scalers["dayback_scaler"]
            self.target_scaler = scalers["target_scaler"]

        except Exception as e:
            logging.error(f"Model loading error: {str(e)}")
            logging.error(f"Error type: {type(e).__name__}")
            logging.error("Full traceback:")
            logging.error(traceback.format_exc())
            raise RuntimeError(f"Failed to load model or scalers: {str(e)}")

    # 这里要检查sql是否正确
    def _query_influx(self):
        query = f"""
        from(bucket: "{INFLUX_CONFIG['bucket']}")
            |> range(start: -{PREDICTION_CONFIG['lookback_minutes']}m)
            |> pivot(rowKey: ["_time", "node_id"], columnKey: ["_field"], valueColumn: "_value")
        """

        try:
            result = self.influx_client.query_api().query_data_frame(query)

            if result.empty:
                raise ValueError("No data found in InfluxDB")

            logging.info(f"Query successful, shape: {result.shape}")
            logging.info("Columns:", result.columns.tolist())

            required_columns = [
                "_time",
                "minute_time",
                "node_id",
                "job_count",
                "req_cpu_rate",
                "avg_req_cpu_per_job",
                "avg_job_runtime",
            ]

            for col in required_columns:
                if col not in result.columns:
                    raise ValueError(f"Missing required column: {col}")

            cluster_metrics = result.groupby("minute_time", as_index=False).agg(
                {
                    "job_count": "sum",
                    "req_cpu_rate": "mean",
                    "avg_req_cpu_per_job": "mean",
                    "avg_job_runtime": "mean",
                }
            )

            logging.info(f"cluster_metrics: {cluster_metrics}")

            logging.info("\n=== cluster_metrics info ===")
            logging.info(f"Shape: {cluster_metrics.shape}")

            active_nodes_df = result[result["job_count"] > 0]
            logging.info("\n=== active nodes data ===")
            logging.info(f"Active nodes data shape: {active_nodes_df.shape}")

            active_nodes = (
                result[result["job_count"] > 0]
                .groupby("minute_time")["node_id"]
                .nunique()
                .reindex(cluster_metrics["minute_time"])
                .fillna(0)
            )

            logging.info("\n=== active_nodes info ===")
            logging.info(f"active_nodes shape: {active_nodes.shape}")
            logging.info(f"Current active nodes: {active_nodes.values[-1]}")  

            df = pd.DataFrame()
            df["datetime"] = pd.to_datetime(cluster_metrics["minute_time"], unit="s")
            df["active_node_count"] = active_nodes.values
            df["running_job_count"] = cluster_metrics["job_count"].fillna(0)
            df["avg_req_cpu_rate"] = cluster_metrics["req_cpu_rate"].fillna(
                0
            )
            df["avg_req_cpu_per_job"] = cluster_metrics["avg_req_cpu_per_job"].fillna(
                0
            )
            df["avg_runtime_minutes"] = cluster_metrics["avg_job_runtime"].fillna(0)

            logging.info(f"df: {df}")
            return df

        except Exception as e:
            logging.error(f"Query error: {str(e)}")
            raise

    def predict(self, avg_req_node_per_job, total_nodes):
        try:
            # 1. 获取数据
            logging.info("\n=== 开始预测流程 ===")
            logging.info(f"输入参数: avg_req_node_per_job={avg_req_node_per_job}, total_nodes={total_nodes}")
            
            df = self._query_influx()
            logging.info("\n原始数据概览:")
            logging.info(f"DataFrame shape: {df.shape}")
            logging.info(f"DataFrame head:\n{df.head()}")
            logging.info(f"DataFrame tail:\n{df.tail()}")
            
            # 检查数据是否为空或包含NaN
            logging.info("\n数据质量检查:")
            logging.info(f"是否存在空值: {df.isnull().any().any()}")
            if df.isnull().any().any():
                logging.info("空值统计:\n" + df.isnull().sum().to_string())

            df["avg_req_node_per_job"] = avg_req_node_per_job

            # 2. 处理数据
            past_hour_data, cur_datetime, dayback_data = self._process_data(df, total_nodes)

            logging.info("\n处理后的数据维度:")
            logging.info(f"past_hour_data shape: {past_hour_data.shape}")
            logging.info(f"past_hour_data 范围: [{past_hour_data.min()}, {past_hour_data.max()}]")
            logging.info(f"past_hour_data 样本:\n{past_hour_data[:5]}")
            
            logging.info(f"\ncur_datetime shape: {cur_datetime.shape}")
            logging.info(f"cur_datetime 值: {cur_datetime}")
            
            logging.info(f"\ndayback_data shape: {dayback_data.shape}")
            logging.info(f"dayback_data 值: {dayback_data}")

            # 3. 转换为张量
            self.model.eval()
            with torch.no_grad():
                past_hour_tensor = torch.FloatTensor(past_hour_data).unsqueeze(0).to(self.device)
                cur_datetime_tensor = torch.FloatTensor(cur_datetime).unsqueeze(0).to(self.device)
                dayback_tensor = torch.FloatTensor(dayback_data).unsqueeze(0).to(self.device)
                
                logging.info("\n张量维度和值:")
                logging.info(f"past_hour_tensor shape: {past_hour_tensor.shape}")
                logging.info(f"past_hour_tensor 范围: [{past_hour_tensor.min().item()}, {past_hour_tensor.max().item()}]")
                
                logging.info(f"\ncur_datetime_tensor shape: {cur_datetime_tensor.shape}")
                logging.info(f"cur_datetime_tensor 值: {cur_datetime_tensor}")
                
                logging.info(f"\ndayback_tensor shape: {dayback_tensor.shape}")
                logging.info(f"dayback_tensor 值: {dayback_tensor}")
                
                # 4. 模型预测
                logging.info("\n开始模型预测")
                prediction_scaled = self.model(
                    past_hour_tensor, cur_datetime_tensor, dayback_tensor
                )
                
                logging.info(f"缩放后的预测值 shape: {prediction_scaled.shape}")
                logging.info(f"缩放后的预测值: {prediction_scaled.cpu().numpy()}")
                
                # 5. 反向缩放
                prediction = self.target_scaler.inverse_transform(
                    prediction_scaled.cpu().numpy()
                )
                
                logging.info(f"\n反向缩放后的预测值: {prediction}")
                
                # 6. 最终处理
                final_prediction = max(0, round(float(prediction[0][0])))
                logging.info(f"最终预测结果: {final_prediction}")
                
                return final_prediction

        except Exception as e:
            logging.error("\n=== 预测过程出现错误 ===")
            logging.error(f"错误类型: {type(e).__name__}")
            logging.error(f"错误信息: {str(e)}")
            logging.error("完整错误追踪:")
            logging.error(traceback.format_exc())
            return 0

    def _process_data(self, df, total_nodes):
        past_hour_features = df[self.feature_names].values[
            -PREDICTION_CONFIG["lookback_minutes"] :
        ]

        past_hour_features = self.feature_scaler.transform(past_hour_features)

        cur_time = df["datetime"].iloc[-1]
        cur_datetime_features = self._create_time_features(
            cur_time,
            cur_time + timedelta(minutes=PREDICTION_CONFIG["forecast_minutes"]),
        )

        current_idx = len(df) - 1
        dayback_features = self._get_dayback_features(
            df, current_idx, "active_node_count", total_nodes
        )

        return past_hour_features, cur_datetime_features, dayback_features

    def _create_time_features(self, start_time, end_time):
        """Create time range feature vector"""
        is_weekend = float(start_time.dayofweek >= 5)
        is_holiday = float(self._is_holiday(start_time.date()))

        hours = pd.date_range(start_time, end_time, freq="1min").hour
        period_counts = np.zeros(6)
        for hour in hours:
            period = self._get_day_period(hour)
            period_counts[period] += 1

        main_period = np.argmax(period_counts)

        return np.array([
            is_weekend,
            is_holiday,
            main_period / 6.0,
        ], dtype=np.float32)

    def _get_day_period(self, hour):
        """Get period of the day"""
        if 5 <= hour < 9:
            return 0  # early morning
        elif 9 <= hour < 12:
            return 1  # morning
        elif 12 <= hour < 14:
            return 2  # noon
        elif 14 <= hour < 18:
            return 3  # afternoon
        elif 18 <= hour < 24:
            return 4  # evening
        else:
            return 5  # night

    def _is_holiday(self, date):
        return date in holidays.CN()

    def _get_dayback_features(self, df, current_idx, target_col, total_nodes):
        """获取历史模式特征"""
        dayback_features = []

        logging.info("\n=== Dayback Features 处理 ===")
        logging.info(f"当前索引: {current_idx}")

        for days_back in [1, 3, 5, 7]:
            minutes_back = days_back * 24 * 60
            historical_center_idx = current_idx - minutes_back

            if 0 <= historical_center_idx < len(df):
                value = float(df[target_col].iloc[historical_center_idx])
            else:
                value = float(df[target_col].iloc[current_idx])
            
            logging.info(f"{days_back}天前的原始值: {value}")
            dayback_features.append(value)

        dayback_features = np.array(dayback_features, dtype=np.float32)
        
        # 记录归一化前的值
        logging.info(f"归一化前的值: {dayback_features}")
        
        if total_nodes > 0:
            dayback_features = dayback_features / total_nodes
        else:
            dayback_features = np.zeros_like(dayback_features)
        
        # 记录归一化后、缩放前的值
        logging.info(f"归一化后、缩放前的值: {dayback_features}")
        
        # 应用 RobustScaler
        dayback_features = self.dayback_scaler.transform(
            dayback_features.reshape(1, -1)
        )
        
        # 记录最终缩放后的值
        logging.info(f"最终缩放后的值: {dayback_features[0]}")
        
        return dayback_features[0]

    def predict_from_config_data(self):
        """使用config中配置的数据进行预测"""
        try:
            from config import MODEL_CONFIG
            from data_loader import DataLoader
            
            logging.info("\n=== 开始使用配置数据进行预测 ===")
            
            # 初始化数据加载器
            data_loader = DataLoader()
            
            # 加载测试数据
            test_loader = data_loader.create_test_loader(
                batch_size=MODEL_CONFIG['batch_size']
            )
            
            predictions = []
            targets = []
            self.model.eval()
            print("开始预测")
            with torch.no_grad():
                for batch in test_loader:
                    # 获取数据
                    past_hour = batch['past_hour'].to(self.device)
                    cur_datetime = batch['cur_datetime'].to(self.device)
                    dayback = batch['dayback'].to(self.device)
                    target = batch['target']
                    
                    # 模型预测
                    outputs = self.model(past_hour, cur_datetime, dayback)
                    
                    # 记录预测结果
                    print(f"Raw prediction shape: {outputs.shape}")
                    print(f"Raw prediction range: [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                    
                    # 转换预测结果
                    batch_predictions = outputs.cpu().numpy()
                    predictions.extend(batch_predictions)
                    targets.extend(target.numpy())
            
            # 转换为numpy数组并反向缩放
            predictions = np.array(predictions)
            targets = np.array(targets)
            predictions = data_loader.inverse_transform_y(predictions)
            targets = data_loader.inverse_transform_y(targets)
            
            # 确保预测结果为非负整数
            predictions = np.maximum(0, np.round(predictions))
            targets = np.maximum(0, np.round(targets))
            
            # 计算不同误差范围内的准确率
            errors = np.abs(predictions - targets)
            print("\n=== 不同误差范围内的准确率 ===")
            for i in range(1, 11):  # 1-10个节点的误差范围
                accuracy = (errors <= i).mean() * 100
                print(f"误差在{i}个节点以内的准确率: {accuracy:.2f}%")
            
            print("\n=== 预测完成 ===")
            print(f"预测样本数量: {len(predictions)}")
            print(f"预测值范围: [{predictions.min()}, {predictions.max()}]")
            print(f"预测均值: {predictions.mean():.2f}")
            print(f"预测标准差: {predictions.std():.2f}")
            
            return predictions
            
        except Exception as e:
            print("\n=== 预测配置数据时出错 ===")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            print("完整错误追踪:")
            print(traceback.format_exc())
            return None

    def predict_from_csv(self):
        """使用CSV文件数据进行预测"""
        try:
            from config import MODEL_CONFIG
            import pandas as pd
            
            print("\n=== 开始使用CSV数据进行预测 ===")
            
            # 加载CSV数据
            df = pd.read_csv(MODEL_CONFIG['data_path'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)
            
            predictions = []
            targets = []
            total_nodes = MODEL_CONFIG['total_nodes']
            
            # 使用滑动窗口进行预测
            lookback = PREDICTION_CONFIG['lookback_minutes']
            forecast_minutes = PREDICTION_CONFIG['forecast_minutes']

            total_rows = len(df)
            start_idx = int(total_rows * 0.95)  # 从90%位置开始
            df = df.iloc[start_idx:].reset_index(drop=True)
            
            for i in range(lookback, len(df) - forecast_minutes):
                # 获取当前窗口的数据
                window_df = df.iloc[i-lookback:i+1]
                
                # 使用原有的数据处理函数
                past_hour_features, cur_datetime_features, dayback_features = self._process_data(
                    window_df, total_nodes
                )
                
                # 转换为张量
                past_hour_tensor = torch.FloatTensor(past_hour_features).unsqueeze(0).to(self.device)
                cur_datetime_tensor = torch.FloatTensor(cur_datetime_features).unsqueeze(0).to(self.device)
                dayback_tensor = torch.FloatTensor(dayback_features).unsqueeze(0).to(self.device)
                
                # 获取目标值
                target = df.iloc[i+forecast_minutes]['active_node_count']
                
                # 模型预测
                self.model.eval()
                with torch.no_grad():
                    output = self.model(past_hour_tensor, cur_datetime_tensor, dayback_tensor)
                    predictions.append(output.cpu().numpy().reshape(-1)[0])  # 只修改这里确保维度正确
                    targets.append(target)
                
                if i % 100 == 0:
                    print(f"已处理 {i}/{len(df)} 条数据")
            
            # 转换为numpy数组
            predictions = np.array(predictions)
            targets = np.array(targets)
            
            # 在计算误差之前，确保 predictions 和 targets 的维度正确
            predictions = np.array(predictions).reshape(-1)  # 展平成一维数组
            targets = np.array(targets).reshape(-1)         # 展平成一维数组
            
            print(f"\n预测值形状: {predictions.shape}")  # 打印形状以验证
            print(f"目标值形状: {targets.shape}")
            
            # 反向缩放
            predictions = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(-1)
            
            # 确保预测结果为非负整数
            predictions = np.maximum(0, np.round(predictions))
            targets = np.maximum(0, np.round(targets))
            
            # 计算不同误差范围内的准确率
            errors = np.abs(predictions - targets)  # 现在两个数组都是一维的，不会发生广播
            
            print("\n=== 不同误差范围内的准确率 ===")
            for i in range(1, 11):  # 1-10个节点的误差范围
                accuracy = (errors <= i).mean() * 100
                print(f"误差在{i}个节点以内的准确率: {accuracy:.2f}%")
            
            print("\n=== 预测完成 ===")
            print(f"预测样本数量: {len(predictions)}")
            print(f"预测值范围: [{predictions.min()}, {predictions.max()}]")
            print(f"预测均值: {predictions.mean():.2f}")
            print(f"预测标准差: {predictions.std():.2f}")
            
            return predictions, targets
            
        except Exception as e:
            print("\n=== 预测CSV数据时出错 ===")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {str(e)}")
            print("完整错误追踪:")
            print(traceback.format_exc())
            return None, None


app = Flask(__name__)
predictor = NodePredictor()


@app.route("/predict", methods=["POST"])
def predict_nodes():
    try:
        request_data = request.get_json()
        if request_data is None:
            return jsonify({"error": "Request body must be JSON"}), 400
            
        avg_req_node_per_job = request_data.get("avg_nodes_per_job", 0)
        total_nodes = request_data.get("total_nodes", 0)
        logging.info(f"Received prediction request with avg_req_node_per_job: {avg_req_node_per_job}, total_nodes: {total_nodes}")

        prediction = predictor.predict(avg_req_node_per_job, total_nodes)
        logging.info(f"Received prediction request with avg_req_node_per_job: {avg_req_node_per_job}, total_nodes: {total_nodes}")
        logging.info(f"Prediction: {prediction}")
        
        return jsonify({"prediction": prediction})
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        logging.error(f"Error type: {type(e).__name__}")
        logging.error("Full traceback:")
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    if MODE == 'test':
        predictor = NodePredictor()
        # print("\n=== 使用处理好的数据进行预测 ===")
        # predictions_processed = predictor.predict_from_config_data()
        # print("预测完成")
        # print("实际值范围: [{predictions_processed.min()}, {predictions_processed.max()}]")
        # print("实际值均值: {predictions_processed.mean():.2f}")
        # print("实际值标准差: {predictions_processed.std():.2f}")
        
        print("\n=== 使用CSV原始数据进行预测 ===")
        predictions_csv, targets_csv = predictor.predict_from_csv()
        
        if predictions_csv is not None and targets_csv is not None:
            print("\n=== CSV数据预测结果对比 ===")
            print(f"实际值范围: [{targets_csv.min()}, {targets_csv.max()}]")
            print(f"实际值均值: {targets_csv.mean():.2f}")
            print(f"实际值标准差: {targets_csv.std():.2f}")
    else:
        if DEBUG:
            app.run(host="0.0.0.0", port=PORT, debug=True)
        else:
            serve(app, host="0.0.0.0", port=PORT)
