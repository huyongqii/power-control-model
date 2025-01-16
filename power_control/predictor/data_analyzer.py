import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
import os
from datetime import datetime, timedelta
import calendar
import matplotlib.dates as mdates

# 设置seaborn样式
sns.set_theme()  # 使用seaborn的默认主题
sns.set_style("whitegrid")  # 设置网格样式
sns.set_palette("deep")  # 设置颜色主题

# 设置图表样式
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2
})

class DataAnalyzer:
    """HPC cluster data analyzer"""
    
    def __init__(self, data_path: str):
        """Initialize data analyzer"""
        self.cn_holidays = holidays.CN()
        self.data = self._load_data(data_path)
        self.output_dir = os.path.join(os.path.dirname(data_path), 'analysis')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 计算数据的时间范围
        self.duration_days = (self.data['datetime'].max() - self.data['datetime'].min()).days
        print(f"Data spans {self.duration_days} days")
        print(f"Date range: {self.data['datetime'].min()} to {self.data['datetime'].max()}")
            
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load and preprocess data"""
        df = pd.read_csv(data_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Add time-related features
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.day
        df['weekday'] = df['datetime'].dt.weekday
        df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
        df['is_holiday'] = df['datetime'].dt.date.map(
            lambda x: 1 if x in self.cn_holidays else 0
        )
        
        # Add time period classification
        df['time_period'] = pd.cut(
            df['hour'],
            bins=[-1, 5, 8, 11, 13, 17, 19, 23],
            labels=['Dawn', 'Morning', 'AM', 'Noon', 'PM', 'Evening', 'Night']
        )
        
        return df
    
    def analyze_all(self):
        """Run all analyses"""
        # 首先检查数据日期
        self.check_data_dates()
        
        # 然后进行其他分析
        self.plot_daily_patterns()
        self.plot_weekly_patterns()
        self.plot_holiday_comparison()
        self.plot_time_period_patterns()
        self.plot_correlation_heatmap()
        self.plot_job_node_relationship()
        self.plot_utilization_analysis()
        self.plot_monthly_patterns()

    def plot_daily_patterns(self):
        """Analyze and plot daily patterns"""
        plt.figure(figsize=(15, 10))
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # 添加日期信息以区分不同天数
        self.data['date'] = self.data['datetime'].dt.date
        
        # 先计算每天每小时的平均值，然后再计算所有天的平均值
        daily_hourly_avg = self.data.groupby(['date', 'hour']).agg({
            'nb_computing': 'mean',
            'running_jobs': 'mean'
        }).reset_index()
        
        # 计算所有天的每小时平均值
        hourly_avg = daily_hourly_avg.groupby('hour').agg({
            'nb_computing': ['mean', 'std'],
            'running_jobs': ['mean', 'std']
        }).reset_index()
        
        # Plot active nodes with confidence interval
        ax1.plot(hourly_avg['hour'], hourly_avg['nb_computing']['mean'], 
                marker='o', linewidth=2, label='Mean')
        ax1.fill_between(hourly_avg['hour'],
                         hourly_avg['nb_computing']['mean'] - hourly_avg['nb_computing']['std'],
                         hourly_avg['nb_computing']['mean'] + hourly_avg['nb_computing']['std'],
                         alpha=0.2, label='±1 std')
        ax1.set_title('Average Active Nodes per Hour (Full Year)')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Average Active Nodes')
        ax1.legend()
        ax1.grid(True)
        
        # Plot running jobs with confidence interval
        ax2.plot(hourly_avg['hour'], hourly_avg['running_jobs']['mean'], 
                marker='o', linewidth=2, color='orange', label='Mean')
        ax2.fill_between(hourly_avg['hour'],
                         hourly_avg['running_jobs']['mean'] - hourly_avg['running_jobs']['std'],
                         hourly_avg['running_jobs']['mean'] + hourly_avg['running_jobs']['std'],
                         alpha=0.2, color='orange', label='±1 std')
        ax2.set_title('Average Running Jobs per Hour (Full Year)')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Average Running Jobs')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'daily_patterns.png'))
        plt.close()
        
    def plot_weekly_patterns(self):
        """Analyze and plot weekly patterns"""
        plt.figure(figsize=(15, 10))
        
        # Calculate average values per weekday
        weekly_avg = self.data.groupby('weekday').agg({
            'nb_computing': 'mean',
            'running_jobs': 'mean',
            'utilization_rate': 'mean'
        }).reset_index()
        
        # Set x-axis labels
        weekday_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Create three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
        
        # Plot active nodes
        ax1.bar(weekday_labels, weekly_avg['nb_computing'])
        ax1.set_title('Average Active Nodes by Weekday')
        ax1.set_ylabel('Average Active Nodes')
        
        # Plot running jobs
        ax2.bar(weekday_labels, weekly_avg['running_jobs'], color='orange')
        ax2.set_title('Average Running Jobs by Weekday')
        ax2.set_ylabel('Average Running Jobs')
        
        # Plot utilization rate
        ax3.bar(weekday_labels, weekly_avg['utilization_rate'] * 100, color='green')
        ax3.set_title('Average Utilization Rate by Weekday')
        ax3.set_ylabel('Utilization Rate (%)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'weekly_patterns.png'))
        plt.close()
        
    def plot_holiday_comparison(self):
        """Compare holidays vs weekdays"""
        # Calculate average values for different date types
        holiday_stats = self.data.groupby(['is_holiday', 'hour']).agg({
            'nb_computing': 'mean',
            'running_jobs': 'mean',
            'utilization_rate': 'mean'
        }).reset_index()
        
        # Create plots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
        
        # Plot active nodes comparison
        for holiday in [0, 1]:
            data = holiday_stats[holiday_stats['is_holiday'] == holiday]
            label = 'Holiday' if holiday else 'Weekday'
            ax1.plot(data['hour'], data['nb_computing'], 
                    marker='o', label=label)
        ax1.set_title('Holidays vs Weekdays - Active Nodes')
        ax1.set_xlabel('Hour')
        ax1.set_ylabel('Average Active Nodes')
        ax1.legend()
        ax1.grid(True)
        
        # Plot running jobs comparison
        for holiday in [0, 1]:
            data = holiday_stats[holiday_stats['is_holiday'] == holiday]
            label = 'Holiday' if holiday else 'Weekday'
            ax2.plot(data['hour'], data['running_jobs'], 
                    marker='o', label=label)
        ax2.set_title('Holidays vs Weekdays - Running Jobs')
        ax2.set_xlabel('Hour')
        ax2.set_ylabel('Average Running Jobs')
        ax2.legend()
        ax2.grid(True)
        
        # Plot utilization rate comparison
        for holiday in [0, 1]:
            data = holiday_stats[holiday_stats['is_holiday'] == holiday]
            label = 'Holiday' if holiday else 'Weekday'
            ax3.plot(data['hour'], data['utilization_rate'] * 100, 
                    marker='o', label=label)
        ax3.set_title('Holidays vs Weekdays - Utilization Rate')
        ax3.set_xlabel('Hour')
        ax3.set_ylabel('Average Utilization Rate (%)')
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'holiday_comparison.png'))
        plt.close()
        
    def plot_time_period_patterns(self):
        """Analyze patterns across different time periods"""
        # Calculate averages for each time period
        period_stats = self.data.groupby('time_period').agg({
            'nb_computing': 'mean',
            'running_jobs': 'mean',
            'utilization_rate': 'mean'
        }).reset_index()
        
        # Create plots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
        
        # Plot active nodes
        sns.barplot(data=period_stats, x='time_period', y='nb_computing', ax=ax1)
        ax1.set_title('Average Active Nodes by Time Period')
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Average Active Nodes')
        
        # Plot running jobs
        sns.barplot(data=period_stats, x='time_period', y='running_jobs', ax=ax2)
        ax2.set_title('Average Running Jobs by Time Period')
        ax2.set_xlabel('Time Period')
        ax2.set_ylabel('Average Running Jobs')
        
        # Plot utilization rate
        sns.barplot(data=period_stats, x='time_period', 
                   y='utilization_rate', ax=ax3)
        ax3.set_title('Average Utilization Rate by Time Period')
        ax3.set_xlabel('Time Period')
        ax3.set_ylabel('Utilization Rate')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'time_period_patterns.png'))
        plt.close()
        
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap"""
        # Select columns to analyze
        columns = ['nb_computing', 'running_jobs', 'waiting_jobs', 
                  'utilization_rate', 'hour', 'is_weekend', 'is_holiday']
        
        # Set English column name mapping
        column_names = {
            'nb_computing': 'Computing Nodes',
            'running_jobs': 'Running Jobs',
            'waiting_jobs': 'Waiting Jobs',
            'utilization_rate': 'Utilization',
            'hour': 'Hour',
            'is_weekend': 'Is Weekend',
            'is_holiday': 'Is Holiday'
        }
        
        # Calculate correlation matrix
        corr_matrix = self.data[columns].corr()
        
        # Create plot
        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        
        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        
        # Set English labels
        ax.set_xticklabels([column_names[col] for col in columns])
        ax.set_yticklabels([column_names[col] for col in columns])
        ax.set_title('Feature Correlation Heatmap')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_heatmap.png'))
        plt.close()
        
    def plot_job_node_relationship(self):
        """Analyze relationship between job numbers and node numbers"""
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot
        plt.scatter(self.data['running_jobs'], self.data['nb_computing'], 
                   alpha=0.5)
        plt.title('Relationship between Running Jobs and Active Nodes')
        plt.xlabel('Number of Running Jobs')
        plt.ylabel('Number of Active Nodes')
        
        # Add trend line
        z = np.polyfit(self.data['running_jobs'], self.data['nb_computing'], 1)
        p = np.poly1d(z)
        plt.plot(self.data['running_jobs'], p(self.data['running_jobs']), 
                "r--", alpha=0.8)
        
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'job_node_relationship.png'))
        plt.close()
        
    def plot_utilization_analysis(self):
        """Analyze cluster utilization"""
        # Create figure with secondary y-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot utilization distribution histogram
        sns.histplot(data=self.data, x='utilization_rate', bins=50, ax=ax1)
        ax1.set_title('Cluster Utilization Distribution')
        ax1.set_xlabel('Utilization Rate')
        ax1.set_ylabel('Frequency')
        
        # Plot utilization over time with daily ticks
        ax2.plot(self.data['datetime'], self.data['utilization_rate'])
        ax2.set_title('Cluster Utilization Over Time')
        
        # Format x-axis to show days
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=5))  # 每5天显示一个刻度
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 获取实际的开始和结束时间
        start_date = datetime(2025, 1, 1)
        end_date = start_date + timedelta(days=self.duration_days)
        ax2.set_xlim(start_date, end_date)
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Utilization Rate')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'utilization_analysis.png'))
        plt.close()

    def plot_monthly_patterns(self):
        """Analyze and plot monthly patterns"""
        # Add month and year information
        self.data['month'] = self.data['datetime'].dt.month
        self.data['year'] = self.data['datetime'].dt.year
        
        # Calculate monthly averages with year
        monthly_avg = self.data.groupby(['year', 'month']).agg({
            'nb_computing': 'mean',
            'running_jobs': 'mean',
            'utilization_rate': 'mean'
        }).reset_index()
        
        # Create plots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
        
        # Plot monthly active nodes
        for year in sorted(monthly_avg['year'].unique()):
            year_data = monthly_avg[monthly_avg['year'] == year]
            ax1.plot(year_data['month'], year_data['nb_computing'], 
                    marker='o', linewidth=2, label=f'Year {year}')
        ax1.set_title('Average Active Nodes by Month')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Average Active Nodes')
        ax1.set_xticks(range(1, 13))
        ax1.legend()
        ax1.grid(True)
        
        # Plot monthly running jobs (similar changes for year)
        for year in sorted(monthly_avg['year'].unique()):
            year_data = monthly_avg[monthly_avg['year'] == year]
            ax2.plot(year_data['month'], year_data['running_jobs'], 
                    marker='o', linewidth=2, label=f'Year {year}')
        ax2.set_title('Average Running Jobs by Month')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Average Running Jobs')
        ax2.set_xticks(range(1, 13))
        ax2.legend()
        ax2.grid(True)
        
        # Plot monthly utilization (similar changes for year)
        for year in sorted(monthly_avg['year'].unique()):
            year_data = monthly_avg[monthly_avg['year'] == year]
            ax3.plot(year_data['month'], year_data['utilization_rate'], 
                    marker='o', linewidth=2, label=f'Year {year}')
        ax3.set_title('Average Utilization Rate by Month')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Utilization Rate')
        ax3.set_xticks(range(1, 13))
        ax3.legend()
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'monthly_patterns.png'))
        plt.close()

    def check_data_dates(self):
        """检查数据的日期范围和分布"""
        print("\n=== 数据日期检查 ===")
        print(f"数据起始日期: {self.data['datetime'].min()}")
        print(f"数据结束日期: {self.data['datetime'].max()}")
        print(f"总天数: {self.duration_days}")
        
        # 检查每个月的数据点数量
        monthly_counts = self.data.groupby([
            self.data['datetime'].dt.year,
            self.data['datetime'].dt.month
        ]).size()
        
        print("\n每月数据点数量:")
        for (year, month), count in monthly_counts.items():
            print(f"{year}年{month}月: {count}条数据")
        
        # 检查是否有缺失的日期
        date_range = pd.date_range(
            start=self.data['datetime'].min(),
            end=self.data['datetime'].max(),
            freq='H'
        )
        missing_dates = set(date_range) - set(self.data['datetime'])
        if missing_dates:
            print("\n缺失的日期:")
            for date in sorted(list(missing_dates))[:10]:  # 只显示前10个缺失日期
                print(date)
            if len(missing_dates) > 10:
                print(f"... 还有 {len(missing_dates)-10} 个缺失日期")

def main():
    """Main function"""
    # Set data file path
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, 'power_control', 'predictor', 'data',
                            'training_data_20250116_141434.csv')
    
    # Create analyzer and run analysis
    analyzer = DataAnalyzer(data_path)
    analyzer.analyze_all()
    
    print("Data analysis completed, charts saved to analysis directory")

if __name__ == "__main__":
    main() 