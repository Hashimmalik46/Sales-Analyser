import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analysis_engine import UniversalSalesAnalyzer, suggest_mappings, detect_data_types

# Page setup
st.set_page_config(page_title="Advanced Sales Analyzer", layout="wide", initial_sidebar_state="expanded")
st.title("🚀 Advanced Sales Analyzer")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'mapping' not in st.session_state:
    st.session_state.mapping = {}
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
# --- THIS IS THE KEY FIX ---
# Add a new state variable to track the loaded file name
if 'current_file_name' not in st.session_state:
    st.session_state.current_file_name = None

# Main app
tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload Data", "🗺️ Smart Mapping", "📊 Comprehensive Analysis", "📈 Advanced Analytics"])

# TAB 1: Upload Data
with tab1:
    st.header("Step 1: Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload ANY sales data CSV", type=['csv'], help="Supports ANY column names and structure")
    
    if uploaded_file is not None:
        
        # --- THIS IS THE KEY FIX ---
        # Check if this is a NEWLY uploaded file, not just a script re-run
        if uploaded_file.name != st.session_state.current_file_name:
            st.info("New file detected, loading data...")
            try:
                # This is a NEW file. Reset everything.
                st.session_state.df = None
                st.session_state.mapping = {}
                st.session_state.analyzer = None
                st.session_state.current_file_name = uploaded_file.name # Store the new name

                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                st.session_state.df = None # Ensure state is cleared on error
                st.session_state.current_file_name = None # Reset on error
        
        # This part now runs every time (if a file is loaded),
        # but the data is only processed once (above)
        if st.session_state.df is not None:
            st.success(f"✅ Dataset loaded! {st.session_state.df.shape[0]:,} records, {st.session_state.df.shape[1]} columns")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                with st.expander("🔍 Data Preview", expanded=True):
                    st.dataframe(st.session_state.df.head(15), use_container_width=True)
            
            with col2:
                with st.expander("📋 Dataset Info"):
                    df = st.session_state.df # Get from state
                    st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
                    st.write("**Columns & Data Types:**")
                    info_df = pd.DataFrame({
                        "Column": df.columns,
                        "Data Type": [str(dt) for dt in df.dtypes],
                        "Unique Values": df.nunique()
                    }).set_index("Column")
                    st.dataframe(info_df, use_container_width=True)
                    
                    # Data type detection
                    detection = detect_data_types(df)
                    st.write("**Auto-detected Types:**")
                    for dtype, cols in detection.items():
                        if cols:
                            st.write(f"• **{dtype.title()}:** {', '.join(cols)}")
        
    elif st.session_state.current_file_name is not None:
        # --- THIS IS THE KEY FIX ---
        # User cleared the file (uploaded_file is None), so reset everything
        st.session_state.df = None
        st.session_state.mapping = {}
        st.session_state.analyzer = None
        st.session_state.current_file_name = None


# TAB 2: Smart Mapping
with tab2:
    st.header("Step 2: Smart Column Mapping")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Auto-suggest mappings
        suggestions = suggest_mappings(df)
        
        st.subheader("💡 Smart Suggestions")
        st.info("We automatically detected these potential mappings. Please review and correct them below.")
        
        # Display suggestions in a nice format
        suggestion_cols = st.columns(4)
        col_idx = 0
        
        for dim, suggested_cols in list(suggestions.items())[:12]:  # Show first 12
            with suggestion_cols[col_idx % 4]:
                with st.container(border=True):
                    st.markdown(f"**{dim.upper()}**")
                    for col in suggested_cols[:3]:  # Show top 3 suggestions
                        st.write(f"• `{col}`")
                    if len(suggested_cols) > 3:
                        st.write(f"• ... and {len(suggested_cols) - 3} more")
            col_idx += 1
        
        st.markdown("---")
        st.subheader("🗺️ Manual Mapping (Override if needed)")
        
        # Organized mapping interface
        mapping_categories = {
            "💰 Financial Metrics (Required)": ['value', 'quantity', 'cost', 'profit', 'discount', 'shipping', 'tax'],
            "🕒 Time Dimensions (Required for Trends)": ['date', 'year', 'month', 'quarter'],
            "📦 Product Info": ['product', 'product_id', 'category', 'subcategory', 'brand'],
            "👥 Customer Data": ['customer', 'customer_id', 'customer_type'],
            "🌍 Locations": ['region', 'country', 'state', 'city'],
            "📋 Transaction Details": ['order_id', 'salesperson', 'channel', 'payment_method'],
            "⭐ Additional": ['rating']
        }
        
        mapping = {}
        
        for category, dimensions in mapping_categories.items():
            st.markdown(f"### {category}")
            cols = st.columns(3)
            
            for i, dim in enumerate(dimensions):
                with cols[i % 3]:
                    # Find the best default suggestion
                    default_index = 0
                    options = [''] + list(df.columns) # Add '' as the first (empty) option
                    
                    if dim in suggestions:
                        # Prioritize the first suggestion
                        best_guess = suggestions[dim][0]
                        if best_guess in options:
                            default_index = options.index(best_guess)

                    selected_col = st.selectbox(
                        f"{dim.replace('_', ' ').title()}",
                        options=options,
                        index=default_index,
                        key=f"map_{dim}"
                    )
                    
                    if selected_col: # Only add to mapping if user selected something
                        mapping[dim] = selected_col
        
        if st.button("🚀 Initialize Universal Analyzer", type="primary", use_container_width=True):
            if 'date' not in mapping:
                st.warning("⚠️ No 'Date' column mapped. Time series and trend analysis will be disabled.")
            
            if any(dim in mapping for dim in ['value', 'quantity', 'cost']):  # At least one metric
                try:
                    st.session_state.mapping = mapping
                    # Pass a copy of the dataframe
                    analyzer = UniversalSalesAnalyzer(st.session_state.df.copy(), mapping)
                    st.session_state.analyzer = analyzer
                    st.success("🎉 Universal Analyzer Ready! Proceed to analysis tabs.")
                    st.balloons()
                    if 'profit' not in mapping and ('value' in mapping and 'cost' in mapping):
                        st.info("💡 **Auto-Profit Calculated!** You provided 'Value' and 'Cost', so 'Profit' has been calculated automatically.")
                except Exception as e:
                    st.error(f"❌ Error during initialization: {str(e)}")
                    st.session_state.analyzer = None
            else:
                st.error("🚨 Please map at least one financial metric (e.g., Value, Quantity, or Cost) to proceed.")

# TAB 3: Comprehensive Analysis
with tab3:
    st.header("Step 3: Comprehensive Analysis")
    
    # This check now works correctly
    if st.session_state.analyzer is not None:
        analyzer = st.session_state.analyzer
        
        # Overview
        st.subheader("📋 Dataset Overview")
        overview = analyzer.get_dataset_overview()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{overview['total_records']:,}")
        with col2:
            st.metric("Total Columns (Standardized)", overview['total_columns'])
        with col3:
            st.metric("Available Dimensions", len(overview['available_dimensions']))
        with col4:
            if overview['date_range'] and overview['date_range']['days'] is not None:
                st.metric("Date Range (Days)", f"{overview['date_range']['days']:,}")
            else:
                st.metric("Date Range (Days)", "N/A")

        
        # Statistics
        st.subheader("📊 Comprehensive Statistics")
        stats = analyzer.get_comprehensive_statistics()
        
        metric_cols = st.columns(len(stats) if stats else 1)
        col_idx = 0
        
        if not stats:
            st.warning("No numeric metrics found or calculated.")
        
        for metric, values in stats.items():
            with metric_cols[col_idx]:
                with st.container(border=True):
                    st.markdown(f"<h5 style='text-align: center;'>{metric.upper()}</h5>", unsafe_allow_html=True)
                    is_currency = any(c in metric for c in ['value', 'cost', 'profit', 'shipping', 'tax'])
                    prefix = "$" if is_currency else ""
                    
                    st.metric("Total", f"{prefix}{values['total']:,.2f}")
                    st.metric("Average", f"{prefix}{values['mean']:,.2f}")
                    st.metric("Median", f"{prefix}{values['median']:,.2f}")
                    st.metric("Count", f"{values['count']:,}")
            col_idx += 1
        
        # Flexible Analysis
        st.subheader("🔍 Custom Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            # Filter out non-categorical dimensions for grouping
            grouping_options = [
                dim for dim in analyzer.available_dimensions.keys() 
                if dim not in analyzer.numeric_metrics and dim != 'date'
            ]
            
            if not grouping_options:
                st.warning("No categorical dimensions (like 'category', 'region', 'product') are mapped for grouping.")
                group_dims = []
            else:
                default_group = ['category'] if 'category' in grouping_options else [grouping_options[0]]
                group_dims = st.multiselect(
                    "Group By Dimensions",
                    options=grouping_options,
                    default=default_group
                )
        with col2:
            metric = st.selectbox(
                "Analysis Metric",
                options=analyzer.numeric_metrics,
                index=0,
                key="custom_metric"
            )
        with col3:
            agg_func = st.selectbox(
                "Aggregation",
                options=['sum', 'mean', 'count', 'median', 'max', 'min'],
                index=0,
                key="custom_agg"
            )
        
        if group_dims and metric:
            try:
                result = analyzer.analyze_by_dimensions(group_dims, metric, agg_func)
                if result is not None and not result.empty:
                    st.dataframe(result.head(1000), use_container_width=True, height=300)
                    
                    # Visualization
                    if len(group_dims) == 1:
                        st.markdown("---")
                        st.subheader(f"Visual: {metric.title()} by {group_dims[0].title()}")
                        # Show top N for clarity
                        top_n = result.head(20)
                        fig = px.bar(top_n, x=group_dims[0], y=metric, title=f"Top 20: {metric.title()} by {group_dims[0].title()}")
                        st.plotly_chart(fig, use_container_width=True)
                    elif len(group_dims) == 2:
                        st.markdown("---")
                        st.subheader(f"Visual: {metric.title()} by {group_dims[0].title()} and {group_dims[1].title()}")
                        # Use a pivot for heatmap or grouped bar
                        try:
                            pivot_data = result.pivot(index=group_dims[0], columns=group_dims[1], values=metric).fillna(0)
                            st.dataframe(pivot_data, use_container_width=True)
                            fig = px.imshow(pivot_data, aspect="auto", title=f"{metric.title()} Heatmap")
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.info(f"Could not create pivot/heatmap (likely too many unique values): {e}")
                else:
                    st.info("No data to display for this combination.")
            except Exception as e:
                st.error(f"Error during custom analysis: {e}")
    
    else:
        # This is the warning message
        st.warning("Please upload a dataset and initialize the analyzer in '📁 Upload Data' and '🗺️ Smart Mapping' tabs first.")


# TAB 4: Advanced Analytics
with tab4:
    st.header("Step 4: Advanced Analytics")
    
    # This check now works correctly
    if st.session_state.analyzer is not None:
        analyzer = st.session_state.analyzer
        
        analysis_type = st.selectbox("Choose Advanced Analysis", [
            "🚀 Complete Report (Recommended)",
            "📈 Time Series Analysis",
            "🔗 Correlation Matrix", 
            "📊 Pivot Tables",
            "👥 Customer Segmentation (RFM)",
            "🌍 Geographic Analysis",
            "📅 Seasonality Patterns",
            "📦 Product Portfolio"
        ])
        
        st.markdown("---")
        
        # --- COMPLETE REPORT ---
        if analysis_type == "🚀 Complete Report (Recommended)":
            st.subheader("Comprehensive Analysis Report")
            
            if st.button("Generate Complete Report", type="primary"):
                with st.spinner("Generating comprehensive analysis... This may take a moment."):
                    report = analyzer.get_comprehensive_report()
                    
                    # 1. Overview
                    with st.expander("📄 OVERVIEW", expanded=True):
                        st.markdown("#### Key Dataset Metrics")
                        overview = report.get('overview', {})
                        o_col1, o_col2, o_col3 = st.columns(3)
                        o_col1.metric("Total Records", f"{overview.get('total_records', 0):,}")
                        o_col2.metric("Available Dimensions", len(overview.get('available_dimensions', [])))
                        if overview.get('date_range') and overview['date_range'].get('days') is not None:
                            o_col3.metric("Date Range (Days)", f"{overview['date_range']['days']:,}")
                        
                        st.markdown("#### Available Dimensions & Metrics")
                        st.json(report.get('available_dimensions', {}))

                    # 2. Statistics
                    with st.expander("📊 KEY FINANCIAL STATISTICS", expanded=True):
                        stats = report.get('statistics', {})
                        if stats:
                            s_cols = st.columns(len(stats))
                            i = 0
                            for metric, values in stats.items():
                                with s_cols[i]:
                                    is_currency = any(c in metric for c in ['value', 'cost', 'profit'])
                                    prefix = "$" if is_currency else ""
                                    s_cols[i].metric(f"Total {metric.title()}", f"{prefix}{values['total']:,.2f}")
                                    s_cols[i].metric(f"Average {metric.title()}", f"{prefix}{values['mean']:,.2f}")
                                i += 1
                        else:
                            st.info("No financial statistics available.")
                    
                    # 3. Trends
                    with st.expander("📈 SALES TRENDS OVER TIME"):
                        trends = report.get('trends', {})
                        monthly_trend = trends.get('monthly')
                        if monthly_trend is not None and not monthly_trend.empty:
                            st.markdown("#### Monthly Sales Trend ('Value')")
                            fig = px.line(monthly_trend, x='period', y='sum', title="Monthly Sales (Value) Over Time", markers=True)
                            fig.update_layout(xaxis_title="Month", yaxis_title="Total Sales Value")
                            st.plotly_chart(fig, use_container_width=True)
                            st.dataframe(monthly_trend, use_container_width=True)
                        else:
                            st.info("No 'Date' column mapped. Cannot generate trend analysis.")
                    
                    # 4. Correlations
                    with st.expander("🔗 CORRELATION MATRIX"):
                        corr_matrix = report.get('correlations')
                        if corr_matrix is not None and not corr_matrix.empty:
                            st.markdown("#### Correlation Between Numeric Metrics")
                            fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Matrix", aspect="auto",
                                            color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Not enough numeric columns (need at least 2) for a correlation matrix.")
                            
                    # 5. Segmentation
                    with st.expander("👥 CUSTOMER & PRODUCT SEGMENTATION"):
                        segmentation = report.get('segmentation', {})
                        
                        st.markdown("#### Customer RFM Segmentation")
                        rfm = segmentation.get('rfm')
                        if rfm is not None and not rfm.empty:
                            st.info("RFM (Recency, Frequency, Monetary) analysis of top customers.")
                            st.dataframe(rfm.head(20), use_container_width=True)
                        else:
                            st.info("Could not perform RFM analysis. Requires 'customer', 'date', and 'value' columns.")
                        
                        st.markdown("#### Product Performance")
                        products = segmentation.get('products')
                        if products is not None and not products.empty:
                            st.info("Top performing products by total sales value.")
                            st.dataframe(products.sort_values('total_sales', ascending=False).head(20), use_container_width=True)
                        else:
                            st.info("Could not perform product analysis. Requires 'product' and 'value' columns.")

                    # 6. Geography
                    with st.expander("🌍 GEOGRAPHIC ANALYSIS"):
                        geography = report.get('geography', {})
                        if geography:
                            st.markdown("#### Sales by Location")
                            geo_dim = list(geography.keys())[0] # Get first available geo dimension
                            geo_data = geography[geo_dim].reset_index().head(20)
                            geo_data = geo_data.sort_values(f'total_sales', ascending=False)
                            
                            st.info(f"Showing Top 20 by '{geo_dim.title()}'")
                            fig = px.bar(geo_data, x=geo_dim, y='total_sales', title=f"Top 20 Sales by {geo_dim.title()}")
                            st.plotly_chart(fig, use_container_width=True)
                            st.dataframe(geo_data, use_container_width=True)
                        else:
                            st.info("No location columns ('region', 'country', 'state', 'city') mapped.")

                    # 7. Seasonality
                    with st.expander("📅 SEASONALITY PATTERNS"):
                        seasonality = report.get('seasonality', {})
                        if seasonality:
                            st.markdown("#### Sales by Day of Week")
                            daily = seasonality.get('daily')
                            if daily is not None and not daily.empty:
                                fig = px.bar(daily.reset_index(), x='day_of_week', y='sum', title="Sales by Day of Week",
                                             category_orders={"day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]})
                                st.plotly_chart(fig, use_container_width=True)
                                
                            st.markdown("#### Sales by Month")
                            monthly = seasonality.get('monthly')
                            if monthly is not None and not monthly.empty:
                                fig = px.bar(monthly.reset_index(), x='month', y='sum', title="Sales by Month")
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No 'Date' column mapped. Cannot generate seasonality analysis.")
        
        # --- TIME SERIES ---
        elif analysis_type == "📈 Time Series Analysis":
            st.subheader("Time Series Analysis")
            if 'date' not in analyzer.available_dimensions:
                st.warning("A 'Date' column must be mapped in Step 2 for time series analysis.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    freq = st.selectbox("Frequency", 
                                        options={'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly', 'Q': 'Quarterly', 'Y': 'Yearly'},
                                        index=2,
                                        format_func=lambda x: f"{x} - { {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly', 'Q': 'Quarterly', 'Y': 'Yearly'}[x]}")
                with col2:
                    metric = st.selectbox("Metric", analyzer.numeric_metrics, index=0)
                
                time_data = analyzer.get_time_series_analysis(freq, metric)
                if time_data is not None and not time_data.empty:
                    fig = px.line(time_data, x='period', y='sum', title=f"{metric.title()} Over Time ({freq})", markers=True)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("Show Raw Time Series Data"):
                        st.dataframe(time_data, use_container_width=True)
                else:
                    st.error("Could not generate time series data.")
        
        # --- CORRELATION ---
        elif analysis_type == "🔗 Correlation Matrix":
            st.subheader("Correlation Analysis")
            corr_matrix = analyzer.get_correlation_matrix()
            if corr_matrix is not None and not corr_matrix.empty:
                fig = px.imshow(corr_matrix, title="Correlation Matrix of Numeric Metrics", text_auto=True,
                                color_continuous_scale='RdBu_r', zmin=-1, zmax=1, aspect="auto")
                st.plotly_chart(fig, use_container_width=True)
                with st.expander("Show Raw Correlation Data"):
                    st.dataframe(corr_matrix, use_container_width=True)
            else:
                st.warning("Not enough numeric columns (need at least 2) for a correlation matrix.")
        
        # --- PIVOT TABLE ---
        elif analysis_type == "📊 Pivot Tables":
            st.subheader("Flexible Pivot Analysis")
            
            categorical_dims = [d for d in analyzer.available_dimensions.keys() if d not in analyzer.numeric_metrics and d != 'date']
            
            if len(categorical_dims) < 2:
                st.warning("You need to map at least two categorical dimensions (e.g., 'Category', 'Region') for a pivot table.")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    index_dim = st.selectbox("Rows (Index)", categorical_dims, index=0)
                with col2:
                    # Ensure columns dim is different from index
                    col_options = [d for d in categorical_dims if d != index_dim]
                    columns_dim = st.selectbox("Columns", col_options, index=0)
                with col3:
                    values_dim = st.selectbox("Values (Metric)", analyzer.numeric_metrics, index=0)
                
                pivot = analyzer.get_pivot_analysis(index_dim, columns_dim, values_dim)
                if pivot is not None:
                    st.dataframe(pivot, use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("Pivot Heatmap")
                    fig = px.imshow(pivot, title=f"{values_dim.title()} by {index_dim.title()} and {columns_dim.title()}", aspect="auto")
                    st.plotly_chart(fig, use_container_width=True)

        # --- OTHER ANALYSES (as standalone tabs) ---
        elif analysis_type == "👥 Customer Segmentation (RFM)":
            st.subheader("Customer Segmentation (RFM)")
            segmentation = analyzer.get_segmentation_analysis()
            rfm = segmentation.get('rfm')
            if rfm is not None and not rfm.empty:
                st.info("RFM (Recency, Frequency, Monetary) analysis. Recency = days since last purchase. Frequency = number of transactions. Monetary = total value spent.")
                st.dataframe(rfm.sort_values('monetary', ascending=False), use_container_width=True)
            else:
                st.warning("Could not perform RFM analysis. Requires 'customer', 'date', 'value', and 'order_id' (for frequency) columns to be mapped.")

        elif analysis_type == "🌍 Geographic Analysis":
            st.subheader("Geographic Analysis")
            geography = analyzer.get_geographic_analysis()
            if geography:
                geo_dim = st.selectbox("Select Geographic Dimension", list(geography.keys()))
                if geo_dim:
                    geo_data = geography[geo_dim].reset_index()
                    st.dataframe(geo_data, use_container_width=True)
                    
                    fig = px.bar(geo_data.head(25), x=geo_dim, y='total_sales', title=f"Top 25 Sales by {geo_dim.title()}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No location columns ('region', 'country', 'state', 'city') mapped.")

        elif analysis_type == "📅 Seasonality Patterns":
            st.subheader("Seasonality Patterns")
            seasonality = analyzer.get_seasonality_analysis()
            if seasonality:
                st.markdown("#### Sales by Day of Week")
                daily = seasonality.get('daily')
                if daily is not None and not daily.empty:
                    fig = px.bar(daily.reset_index(), x='day_of_week', y='sum', title="Sales by Day of Week",
                                 category_orders={"day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]})
                    st.plotly_chart(fig, use_container_width=True)
                    
                st.markdown("#### Sales by Month")
                monthly = seasonality.get('monthly')
                if monthly is not None and not monthly.empty:
                    fig = px.bar(monthly.reset_index(), x='month', y='sum', title="Sales by Month")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No 'Date' column mapped. Cannot generate seasonality analysis.")

        elif analysis_type == "📦 Product Portfolio":
            st.subheader("Product Portfolio Analysis")
            portfolio = analyzer.get_product_portfolio_analysis()
            if portfolio is not None and not portfolio.empty:
                st.info("This analysis segments products based on their total sales (Market Share) and transaction count (Growth Proxy).")
                fig = px.scatter(portfolio, x='market_share', y='growth', 
                                 title="Product Portfolio (BCG Matrix Proxy)",
                                 size='frequency', hover_name=portfolio.index,
                                 color='segment',
                                 labels={'market_share': 'Total Sales Value (Market Share Proxy)', 'growth': 'Total Transactions (Growth Proxy)'})
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("Show Raw Portfolio Data"):
                    st.dataframe(portfolio, use_container_width=True)
            else:
                st.warning("Could not perform portfolio analysis. Requires 'product', 'value', and 'quantity' columns.")

    else:
        # This is the warning message that shows if the analyzer isn't ready
        st.warning("Please upload a dataset and initialize the analyzer in '📁 Upload Data' and '🗺️ Smart Mapping' tabs first.")