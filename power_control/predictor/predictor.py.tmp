import os
import torch
import numpy as np
import pandas as pd
import joblib
import holidays
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient
from model import NodePredictorNN
from config import MODEL_CONFIG

class NodePredictor:
    def __init__(self):
        """Initialize predictor, load pre-trained model and scalers"""
        self.config = MODEL_CONFIG
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cn_holidays = holidays.CN()
        
        self.influx_client = InfluxDBClient(
            url=self.config['influx_url'],
            token=self.config['influx_token'],
            org=self.config['influx_org']
        )
        
        self._init_model()
    
    def _init_model(self):
        """Initialize and load the model and scalers"""
        self.model = NodePredictorNN(feature_size=7).to(self.device)
        
        try:
            checkpoint = torch.load(
                f"{self.config['model_dir']}/checkpoint.pth",
                map_location=self.device,
                weights_only=True
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            scaler_path = os.path.join(self.config['model_dir'], "dataset_scalers.pkl")
            scalers = joblib.load(scaler_path)
            self.feature_scaler = scalers['feature_scaler']
            self.target_scaler = scalers['target_scaler']
            self.dayback_scaler = scalers['dayback_scaler']
            
            print("Successfully loaded pre-trained model and scalers")
        except Exception as e:
            raise RuntimeError(f"Failed to load model or scalers: {str(e)}")
        
        self.model.eval()

    def _query_influx(self, start_time, end_time):
        """
        Query data from InfluxDB and calculate cluster-wide metrics
        
        Args:
            start_time: Start time for the query
            end_time: End time for the query
            
        Returns:
            pd.DataFrame: DataFrame with calculated cluster metrics
        """
        query = f'''
        from(bucket: "{self.config['influx_bucket']}")
            |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
            |> pivot(rowKey:["_time", "node_id"], columnKey: ["_field"], valueColumn: "_value")
        '''
        
        result = self.influx_client.query_api().query_data_frame(query)
        
        if result.empty:
            raise ValueError("No data found in InfluxDB for the specified time range")
        
        # Group by time and calculate cluster-wide metrics
        cluster_metrics = result.groupby('_time').agg({
            'job_count': 'sum',                    # Total running jobs across all nodes
            'cpu_request_ratio': 'mean',           # Average CPU request ratio
            'avg_cpu_per_job': 'mean',            # Average CPUs per job
            'avg_job_runtime': 'mean',            # Average job runtime
            'node_id': 'count'                    # Number of active nodes
        }).reset_index()
        
        # Calculate cluster-wide metrics
        df = pd.DataFrame()
        df['datetime'] = pd.to_datetime(cluster_metrics['_time'])
        
        # Core metrics needed by the model
        df['running_jobs'] = cluster_metrics['job_count']
        df['nb_computing'] = cluster_metrics['node_id']  # Number of active nodes
        df['avg_req_cpu_occupancy_rate'] = cluster_metrics['cpu_request_ratio']
        df['avg_cpus_per_job'] = cluster_metrics['avg_cpu_per_job']
        df['avg_runtime_minutes'] = cluster_metrics['avg_job_runtime']
        
        # Calculate derived metrics
        df['avg_nodes_per_job'] = df['avg_cpus_per_job'] / self.config['cpus_per_node']
        df['waiting_jobs'] = 0  # This needs to be obtained from somewhere else
        
        return df[['datetime', 'running_jobs', 'waiting_jobs', 'nb_computing',
                  'avg_req_cpu_occupancy_rate', 'avg_nodes_per_job',
                  'avg_cpus_per_job', 'avg_runtime_minutes']]

    def predict(self, target_time=None):
        """
        Predict future computing nodes
        
        Args:
            target_time (datetime, optional): Target time for prediction. 
                                           If None, predicts for next forecast_minutes
        
        Returns:
            float: Predicted number of computing nodes
        """
        try:
            current_time = datetime.now()
            if target_time is None:
                target_time = current_time + timedelta(minutes=self.config['forecast_minutes'])
            
            # Calculate required data range
            lookback_minutes = self.config['lookback_minutes']
            start_time = current_time - timedelta(minutes=lookback_minutes)
            
            # Get data from InfluxDB
            df = self._query_influx(start_time, current_time)
            
            # Process data for prediction
            past_hour_data, cur_datetime, dayback_data = self._process_data(df, target_time)
            
            # Convert to tensors and predict
            past_hour_tensor = torch.FloatTensor(past_hour_data).unsqueeze(0).to(self.device)
            cur_datetime_tensor = torch.FloatTensor(cur_datetime).unsqueeze(0).to(self.device)
            dayback_tensor = torch.FloatTensor(dayback_data).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                prediction_scaled = self.model(
                    past_hour_tensor,
                    cur_datetime_tensor,
                    dayback_tensor
                )
            
            prediction = self.target_scaler.inverse_transform(
                prediction_scaled.cpu().numpy()
            )
            
            final_prediction = max(0, round(float(prediction[0][0])))
            
            return final_prediction
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return 0

    def _process_data(self, df, target_time):
        """Process DataFrame into model input format"""
        feature_names = [
            'running_jobs', 'waiting_jobs', 'nb_computing',
            'avg_req_cpu_occupancy_rate', 'avg_nodes_per_job',
            'avg_cpus_per_job', 'avg_runtime_minutes'
        ]
        
        past_hour_features = df[feature_names].values[-self.config['lookback_minutes']:]
        
        cur_time = df['datetime'].iloc[-1]
        cur_datetime_features = self._create_time_features(cur_time, target_time)
        
        timestamps = df['datetime']
        current_idx = len(df) - 1
        dayback_features = self._get_dayback_features(
            df, timestamps, cur_time, current_idx, 'nb_computing'
        )
        
        past_hour_features = self.feature_scaler.transform(past_hour_features)
        dayback_features = self.dayback_scaler.transform(dayback_features.reshape(1, -1))
        
        return past_hour_features, cur_datetime_features, dayback_features

    def _create_time_features(self, start_time, end_time):
        """Create time range feature vector"""
        is_weekend = float(start_time.dayofweek >= 5)
        is_holiday = float(self.is_holiday(start_time.date()))
        
        hours = pd.date_range(start_time, end_time, freq='1min').hour
        period_counts = np.zeros(6)
        for hour in hours:
            period = self.get_day_period(hour)
            period_counts[period] += 1
        
        main_period = np.argmax(period_counts)
        
        return [
            is_weekend,         
            is_holiday,         
            main_period / 6.0,
        ]

    def get_day_period(self, hour):
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

    def is_holiday(self, date):
        """Check if date is holiday"""
        return date in self.cn_holidays

    def _get_dayback_features(self, df, timestamps, target_time, current_idx, target_col):
        """Get historical pattern features"""
        pattern_features = []
        window_minutes = 30
        
        for days_back in [1, 3, 5, 7]:
            minutes_back = days_back * 24 * 60
            historical_center_idx = current_idx - minutes_back
            
            if historical_center_idx >= window_minutes and historical_center_idx + window_minutes < len(df):
                historical_window = df[target_col].iloc[
                    historical_center_idx - window_minutes:
                    historical_center_idx + window_minutes
                ]
                
                pattern_features.extend([
                    float(historical_window.min()),     # min value
                    float(historical_window.max()),     # max value
                ])
            else:
                current_value = float(df[target_col].iloc[current_idx])
                pattern_features.extend([current_value] * 2)
        
        return np.array(pattern_features, dtype=np.float32)
