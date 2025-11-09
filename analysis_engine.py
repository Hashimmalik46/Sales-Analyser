import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class UniversalSalesAnalyzer:
    """
    ULTIMATE FLEXIBLE SALES ANALYZER
    Handles ANY sales dataset with ANY column names
    """
    
    # Define ALL possible standard dimensions for maximum flexibility
    STANDARD_DIMENSIONS = {
        # Core metrics
        'value': ['sales', 'revenue', 'amount', 'price', 'total', 'income', 'value'],
        'quantity': ['quantity', 'qty', 'units', 'volume', 'count'],
        'cost': ['cost', 'cogs', 'expense', 'investment'],
        'profit': ['profit', 'margin', 'gross_profit', 'net_profit'],
        'discount': ['discount', 'discount_amount', 'rebate', 'deduction'],
        
        # Time dimensions
        'date': ['date', 'time', 'datetime', 'timestamp', 'order_date', 'transaction_date'],
        'year': ['year', 'fiscal_year'],
        'month': ['month', 'period'],
        'quarter': ['quarter', 'qtr'],
        
        # Product dimensions
        'product': ['product', 'item', 'sku', 'product_name', 'item_name', 'service'],
        'product_id': ['product_id', 'sku_id', 'item_id'],
        'category': ['category', 'category_name', 'department', 'division', 'type', 'segment'],
        'subcategory': ['subcategory', 'sub_type', 'sub_segment'],
        'brand': ['brand', 'manufacturer', 'vendor', 'supplier'],
        
        # Customer dimensions
        'customer': ['customer', 'client', 'customer_name', 'client_name'],
        'customer_id': ['customer_id', 'client_id'],
        'customer_type': ['customer_type', 'client_type', 'segment'],
        
        # Location dimensions
        'region': ['region', 'territory', 'area', 'zone'],
        'country': ['country', 'nation'],
        'state': ['state', 'province', 'county'],
        'city': ['city', 'town', 'metro'],
        
        # Transaction dimensions
        'order_id': ['order_id', 'transaction_id', 'invoice_id', 'receipt_id'],
        'salesperson': ['salesperson', 'sales_rep', 'agent', 'employee'],
        'channel': ['channel', 'platform', 'store', 'website', 'marketplace'],
        'payment_method': ['payment_method', 'payment_type', 'payment'],
        
        # Additional metrics
        'shipping': ['shipping', 'delivery', 'freight', 'transport'],
        'tax': ['tax', 'vat', 'gst', 'tax_amount'],
        'rating': ['rating', 'score', 'review_score', 'satisfaction']
    }
    
    METRIC_NAMES = ['value', 'quantity', 'cost', 'profit', 'discount', 'shipping', 'tax', 'rating']
    
    def __init__(self, df, column_mapping):
        """
        Initialize with ANY dataset and mapping
        """
        self.original_df = df
        self.mapping = column_mapping
        self.standard_df = self._standardize_dataframe()
        self.available_dimensions = self._get_available_dimensions()
        self.numeric_metrics = [
            dim for dim in self.available_dimensions 
            if dim in self.METRIC_NAMES and pd.api.types.is_numeric_dtype(self.standard_df[dim])
        ]
    
    def _standardize_dataframe(self):
        """Convert to standard column names based on mapping and clean data types"""
        
        # Create a reverse map from {original_col: standard_col}
        reverse_map = {v: k for k, v in self.mapping.items() if v in self.original_df.columns}
        
        # Only select and rename columns that were actually mapped
        standardized_df = self.original_df[list(reverse_map.keys())].rename(columns=reverse_map)
        
        # --- Robust Data Type Conversion ---
        
        # Convert date column
        if 'date' in standardized_df.columns:
            try:
                standardized_df['date'] = pd.to_datetime(standardized_df['date'], errors='coerce')
                # Drop rows where essential date is invalid
                standardized_df = standardized_df.dropna(subset=['date'])
            except Exception as e:
                print(f"Warning: Could not parse date column. {e}")
                # If date parse fails, remove it from processing
                if 'date' in standardized_df.columns:
                     standardized_df = standardized_df.drop(columns=['date'])
        
        # Convert all mapped numeric metrics
        for col in self.METRIC_NAMES:
            if col in standardized_df.columns:
                try:
                    standardized_df[col] = pd.to_numeric(standardized_df[col], errors='coerce')
                except Exception as e:
                    print(f"Warning: Could not convert {col} to numeric. {e}")
        
        # --- Advanced Feature Engineering ---

        # 1. Auto-calculate Profit
        if 'value' in standardized_df.columns and \
           'cost' in standardized_df.columns and \
           'profit' not in standardized_df.columns:
            
            standardized_df['value'] = standardized_df['value'].fillna(0)
            standardized_df['cost'] = standardized_df['cost'].fillna(0)
            standardized_df['profit'] = standardized_df['value'] - standardized_df['cost']
            print("Auto-calculated 'profit' column.")
            
        # 2. Fill NaNs for key metrics for robust aggregation
        for col in ['value', 'quantity', 'cost', 'profit']:
             if col in standardized_df.columns:
                 standardized_df[col] = standardized_df[col].fillna(0)
        
        return standardized_df
    
    def _get_available_dimensions(self):
        """Get list of available dimensions in the standardized data"""
        available = {}
        for dim in self.STANDARD_DIMENSIONS.keys():
            if dim in self.standard_df.columns:
                available[dim] = self.standard_df[dim].nunique()
        
        # Add auto-calculated profit if it exists
        if 'profit' in self.standard_df.columns and 'profit' not in available:
             available['profit'] = self.standard_df['profit'].nunique()
             
        return available
    
    def get_dataset_overview(self):
        """Complete dataset overview"""
        overview = {
            'total_records': len(self.standard_df),
            'total_columns': len(self.standard_df.columns),
            'date_range': None,
            'available_dimensions': self.available_dimensions,
            'data_types': {col: str(dtype) for col, dtype in self.standard_df.dtypes.items()},
            'missing_values': {col: int(count) for col, count in self.standard_df.isnull().sum().items() if count > 0}
        }
        
        if 'date' in self.standard_df.columns:
            min_date = self.standard_df['date'].min()
            max_date = self.standard_df['date'].max()
            overview['date_range'] = {
                'start': min_date.strftime('%Y-%m-%d') if pd.notna(min_date) else None,
                'end': max_date.strftime('%Y-%m-%d') if pd.notna(max_date) else None,
                'days': (max_date - min_date).days if pd.notna(min_date) and pd.notna(max_date) else None
            }
        
        return overview
    
    def get_comprehensive_statistics(self):
        """Statistics for ALL key numeric metrics"""
        stats = {}
        
        for col in self.numeric_metrics:
            stats[col] = {
                'total': float(self.standard_df[col].sum()),
                'mean': float(self.standard_df[col].mean()),
                'median': float(self.standard_df[col].median()),
                'std': float(self.standard_df[col].std()),
                'min': float(self.standard_df[col].min()),
                'max': float(self.standard_df[col].max()),
                'count': int(self.standard_df[col].count()), # Use count() to exclude NaNs
                'missing': int(self.standard_df[col].isnull().sum())
            }
        
        return stats
    
    def analyze_by_dimensions(self, group_by_dims, metric='value', agg_func='sum'):
        """
        Ultra-flexible grouping by ANY dimensions
        """
        if metric not in self.standard_df.columns:
            print(f"Error: Metric '{metric}' not found.")
            return None
        
        available_dims = [dim for dim in group_by_dims if dim in self.standard_df.columns]
        
        if not available_dims:
            print("Error: No valid dimensions to group by.")
            return None
        
        try:
            result = self.standard_df.groupby(available_dims)[metric].agg(agg_func).reset_index()
            result = result.sort_values(metric, ascending=False).round(2)
            return result
        except Exception as e:
            print(f"Error during grouping: {e}")
            return None
    
    def get_time_series_analysis(self, freq='M', metric='value'):
        """Time series analysis for ANY metric using resample"""
        if 'date' not in self.standard_df.columns or metric not in self.standard_df.columns:
            return None
        
        try:
            # Set date as index for resampling
            ts_df = self.standard_df.set_index('date')
            
            # Resample and aggregate
            time_series = ts_df.resample(freq)[metric].agg(['sum', 'mean', 'count']).round(2)
            
            # Format for display
            time_series = time_series.reset_index().rename(columns={'date': 'period'})
            time_series['period'] = time_series['period'].dt.strftime('%Y-%m-%d') # Format period
            return time_series
        except Exception as e:
            print(f"Error in time series analysis: {e}")
            return None
    
    def get_correlation_matrix(self):
        """Correlation between ALL numeric metrics"""
        if len(self.numeric_metrics) < 2:
            return None
        
        corr_df = self.standard_df[self.numeric_metrics]
        return corr_df.corr().round(3)
    
    def get_pivot_analysis(self, index_dim, columns_dim, values_dim='value', aggfunc='sum'):
        """Flexible pivot table analysis"""
        available_dims = [dim for dim in [index_dim, columns_dim, values_dim] if dim in self.standard_df.columns]
        
        if len(available_dims) < 3:
            return None
        
        try:
            pivot_table = self.standard_df.pivot_table(
                values=values_dim,
                index=index_dim,
                columns=columns_dim,
                aggfunc=aggfunc,
                fill_value=0
            ).round(2)
            return pivot_table
        except Exception as e:
            print(f"Error building pivot table: {e}")
            return None
    
    def get_trend_analysis(self):
        """Comprehensive trend analysis"""
        trends = {}
        if 'date' not in self.standard_df.columns:
            return trends
            
        # Monthly trends for 'value'
        monthly = self.get_time_series_analysis('M', 'value')
        if monthly is not None:
            trends['monthly'] = monthly
        
        # Growth rates
        if monthly is not None and not monthly.empty and 'sum' in monthly.columns:
            try:
                monthly['growth_rate'] = monthly['sum'].pct_change().fillna(0)
                trends['growth_rates'] = monthly[['period', 'growth_rate']]
            except Exception as e:
                print(f"Could not calculate growth rates: {e}")
        
        return trends
    
    def get_segmentation_analysis(self):
        """Customer/Product segmentation"""
        segmentation = {}
        
        # --- RFM Analysis ---
        if all(dim in self.standard_df.columns for dim in ['customer', 'date', 'value']):
            try:
                max_date = self.standard_df['date'].max() + pd.Timedelta(days=1)
                
                # Determine frequency column
                if 'order_id' in self.standard_df.columns:
                    freq_agg = ('order_id', 'nunique')
                else:
                    freq_agg = ('date', 'count') # Use transaction count as proxy
                    
                rfm = self.standard_df.groupby('customer').agg(
                    recency=('date', lambda x: (max_date - x.max()).days),
                    frequency=freq_agg,
                    monetary=('value', 'sum')
                ).round(2)
                
                rfm = rfm.sort_values('monetary', ascending=False)
                segmentation['rfm'] = rfm
            except Exception as e:
                print(f"Error during RFM analysis: {e}")
        
        # --- Product performance segments ---
        if 'product' in self.standard_df.columns and 'value' in self.standard_df.columns:
            try:
                product_performance = self.standard_df.groupby('product')['value'].agg(['sum', 'count', 'mean']).round(2)
                product_performance.columns = ['total_sales', 'transaction_count', 'avg_sale_value']
                
                # Segment products
                product_performance['segment'] = pd.qcut(
                    product_performance['total_sales'],
                    q=4,
                    labels=['D-Tier', 'C-Tier', 'B-Tier', 'A-Tier'],
                    duplicates='drop'
                )
                segmentation['products'] = product_performance.sort_values('total_sales', ascending=False)
            except Exception as e:
                print(f"Error during product segmentation: {e}")
        
        return segmentation
    
    def get_geographic_analysis(self):
        """Geographic analysis if location data available"""
        geo_analysis = {}
        if 'value' not in self.standard_df.columns:
            return geo_analysis
            
        location_dims = [dim for dim in ['country', 'state', 'region', 'city'] if dim in self.standard_df.columns]
        
        for loc_dim in location_dims:
            try:
                geo_data = self.standard_df.groupby(loc_dim)['value'].agg(['sum', 'count', 'mean']).round(2)
                geo_data.columns = [f'total_sales', f'transaction_count', f'avg_sale_value']
                geo_analysis[loc_dim] = geo_data.sort_values(f'total_sales', ascending=False)
            except Exception as e:
                print(f"Error in geo analysis for {loc_dim}: {e}")
        
        return geo_analysis
    
    def get_seasonality_analysis(self):
        """Detailed seasonality patterns"""
        if 'date' not in self.standard_df.columns or 'value' not in self.standard_df.columns:
            return None
        
        try:
            seasonal_df = self.standard_df.copy()
            seasonal_df['month'] = seasonal_df['date'].dt.month
            seasonal_df['quarter'] = seasonal_df['date'].dt.quarter
            seasonal_df['day_of_week'] = seasonal_df['date'].dt.day_name()
            
            seasonality = {}
            
            # Monthly patterns
            monthly = seasonal_df.groupby('month')['value'].agg(['sum', 'mean', 'count']).round(2)
            seasonality['monthly'] = monthly
            
            # Quarterly patterns
            quarterly = seasonal_df.groupby('quarter')['value'].agg(['sum', 'mean', 'count']).round(2)
            seasonality['quarterly'] = quarterly
            
            # Day of week patterns
            daily = seasonal_df.groupby('day_of_week')['value'].agg(['sum', 'mean', 'count']).round(2)
            seasonality['daily'] = daily
            
            return seasonality
        except Exception as e:
            print(f"Error in seasonality analysis: {e}")
            return None
    
    def get_product_portfolio_analysis(self):
        """BCG matrix style product analysis"""
        if not all(dim in self.standard_df.columns for dim in ['product', 'value', 'quantity']):
            return None
        
        try:
            portfolio = self.standard_df.groupby('product').agg({
                'value': 'sum',      # Market share proxy
                'quantity': 'sum',   # Growth rate proxy
                'date': 'count'      # Frequency
            }).round(2)
            
            portfolio.columns = ['market_share', 'growth', 'frequency']
            
            # Handle potential case where all values are the same
            if portfolio['growth'].nunique() > 1 and portfolio['market_share'].nunique() > 1:
                portfolio['growth_segment'] = pd.qcut(portfolio['growth'], 2, labels=['Low Growth', 'High Growth'])
                portfolio['share_segment'] = pd.qcut(portfolio['market_share'], 2, labels=['Low Share', 'High Share'])
            
                # Create BCG segments
                portfolio['segment'] = 'Question Mark' # Default
                portfolio.loc[(portfolio['share_segment'] == 'High Share') & (portfolio['growth_segment'] == 'High Growth'), 'segment'] = 'Star'
                portfolio.loc[(portfolio['share_segment'] == 'High Share') & (portfolio['growth_segment'] == 'Low Growth'), 'segment'] = 'Cash Cow'
                portfolio.loc[(portfolio['share_segment'] == 'Low Share') & (portfolio['growth_segment'] == 'Low Growth'), 'segment'] = 'Dog'
            else:
                 portfolio['segment'] = 'N/A' # Not enough variance to segment

            return portfolio.sort_values('market_share', ascending=False)
        except Exception as e:
            print(f"Error in portfolio analysis: {e}")
            return None
    
    def get_comprehensive_report(self):
        """Generate complete analysis report"""
        report = {
            'overview': self.get_dataset_overview(),
            'statistics': self.get_comprehensive_statistics(),
            'correlations': self.get_correlation_matrix(),
            'trends': self.get_trend_analysis(),
            'segmentation': self.get_segmentation_analysis(),
            'geography': self.get_geographic_analysis(),
            'seasonality': self.get_seasonality_analysis(),
            'portfolio': self.get_product_portfolio_analysis(),
            'available_dimensions': self.available_dimensions
        }
        
        return report

# --- Helper Functions ---

def suggest_mappings(df):
    """Automatically suggest column mappings"""
    suggestions = {}
    df_columns = list(df.columns)
    
    for standard_dim, possible_names in UniversalSalesAnalyzer.STANDARD_DIMENSIONS.items():
        matches = []
        for col in df_columns:
            col_lower = col.lower().replace('_', ' ').replace('-', ' ')
            
            # Check if any possible name is in the column name
            for possible in possible_names:
                if possible in col_lower:
                    matches.append(col)
                    break # Go to next column
        
        if matches:
            # Prioritize matches that are more "exact"
            matches_sorted = sorted(matches, key=lambda x: len(x))
            suggestions[standard_dim] = matches_sorted
    
    return suggestions

def detect_data_types(df):
    """Comprehensive data type detection for display"""
    detection = {
        'dates': [],
        'numerics': [],
        'categoricals': [],
        'ids': [],
        'texts': []
    }
    
    for col in df.columns:
        col_lower = col.lower()
        dtype = df[col].dtype
        nunique = df[col].nunique()
        total_rows = len(df)
        
        # 1. Date detection
        if any(keyword in col_lower for keyword in ['date', 'time', 'year', 'month', 'day']):
            detection['dates'].append(col)
        # 2. ID detection
        elif any(keyword in col_lower for keyword in ['id', 'code', 'num', 'no', 'sku']):
            detection['ids'].append(col)
        # 3. Numeric detection (metric-like)
        elif pd.api.types.is_numeric_dtype(dtype) and not any(keyword in col_lower for keyword in ['id', 'year', 'month']):
             detection['numerics'].append(col)
        # 4. Categorical detection (low unique objects)
        elif dtype == 'object' and nunique < min(100, total_rows * 0.5):
            detection['categoricals'].append(col)
        # 5. Text detection (high unique objects)
        elif dtype == 'object':
            detection['texts'].append(col)
        # 6. Fallback for other numerics (like IDs)
        elif pd.api.types.is_numeric_dtype(dtype):
            detection['categoricals'].append(col) # Treat numeric IDs as categoricals
    
    return detection