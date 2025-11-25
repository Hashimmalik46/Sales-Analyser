import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


st.set_page_config(page_title="Sales Analytics Dashboard",
                   layout="wide",
                   page_icon="📊")

st.markdown("""
<style>

body {
    background-color: #0d1117;
    color: #c9d1d9;
}

section.main > div {
    background-color: #0d1117;
}

.sidebar .sidebar-content {
    background-color: #161b22 !important;
}

h1, h2, h3, h4, h5 {
    color: #c9d1d9;
}

.stMetric {
    background-color: #161b22;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #30363d;
}

.block-container {
    padding-top: 1rem;
}

div[data-testid="stMetricValue"] {
    color: #58a6ff !important;
}

div[data-testid="stMetricLabel"] {
    color: #c9d1d9 !important;
}

hr {
    border: 0.5px solid #30363d;
}

</style>
""", unsafe_allow_html=True)


st.title("📊 Sales Analytics Dashboard")


STANDARD_DIMENSIONS = {
    # Core metrics
    'value': [
        'sales', 'revenue', 'amount', 'price', 'total', 'income', 'value',
        'salesamount', 'sales_amt', 'salesvalue', 'discountedprice'
    ],

    'quantity': [
        'quantity', 'qty', 'units', 'volume', 'count',
        'quantitysold', 'qtysold'
    ],

    'cost': [
        'cost', 'cogs', 'expense', 'investment',
        'unitcost'
    ],

    'profit': [
        'profit', 'margin', 'gross_profit', 'net_profit'
    ],

    'discount': [
        'discount', 'discount_amount', 'rebate', 'deduction',
        'discountpercentage'
    ],

    # Time dimensions
    'date': [
        'date', 'time', 'datetime', 'timestamp',
        'order_date', 'transaction_date', 'saledate'
    ],

    'year': ['year', 'fiscal_year'],
    'month': ['month', 'period'],
    'quarter': ['quarter', 'qtr'],

    # Product dimensions
    'product': [
        'product', 'item', 'sku', 'product_name', 'item_name', 'service',
        'productname'
    ],

    'product_id': [
        'product_id', 'sku_id', 'item_id',
        'productid'
    ],

    'category': [
        'category', 'category_name', 'department', 'division', 'type', 'segment',
        'productcategory'
    ],

    'subcategory': ['subcategory', 'sub_type', 'sub_segment'],
    'brand': ['brand', 'manufacturer', 'vendor', 'supplier'],

    # Customer dimensions
    'customer': ['customer', 'client', 'customer_name', 'client_name'],
    'customer_id': ['customer_id', 'client_id'],
    'customer_type': ['customer_type', 'client_type', 'segment'],

    # Location
    'region': ['region', 'territory', 'area', 'zone'],
    'country': ['country', 'nation'],
    'state': ['state', 'province', 'county'],
    'city': ['city', 'town', 'metro'],

    # Transaction dimensions
    'order_id': [
        'order_id', 'transaction_id', 'invoice_id', 'receipt_id'
    ],

    'salesperson': [
        'salesperson', 'sales_rep', 'agent', 'employee',
        'salesrep'
    ],

    'channel': ['channel', 'platform', 'store', 'website', 'marketplace'],
    'payment_method': ['payment_method', 'payment_type', 'payment'],

    # Additional metrics
    'shipping': ['shipping', 'delivery', 'freight', 'transport'],
    'tax': ['tax', 'vat', 'gst', 'tax_amount'],
    'rating': ['rating', 'score', 'review_score', 'satisfaction']
}


PRIORITY_METRICS = ['value', 'quantity', 'profit', 'cost', 'discount']


def normalize_name(s: str) -> str:
    return str(s).lower().replace(" ", "").replace("_", "").replace("-", "")


def detect_column_for_key(df: pd.DataFrame, possible_names: list) -> str | None:
    cols_lower_map = {normalize_name(c): c for c in df.columns}
    for candidate in possible_names:
        key = normalize_name(candidate)
        if key in cols_lower_map:
            return cols_lower_map[key]
    return None


def smart_map_dataframe(df: pd.DataFrame):
    df_copy = df.copy()
    detected = {}

    for std_key, variations in STANDARD_DIMENSIONS.items():
        found = detect_column_for_key(df_copy, variations)
        if found:
            detected[std_key] = found

    rename_map = {orig: std for std, orig in detected.items()}
    return df_copy.rename(columns=rename_map), detected


@st.cache_data(show_spinner=False)
def read_file(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)


st.sidebar.header("📂 Upload & Filters")

# Upload
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
if not uploaded_file:
    st.info("Upload a file to begin.")
    st.stop()

# Read file
try:
    raw_df = read_file(uploaded_file)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

# Mapping
mapped_df, detected_map = smart_map_dataframe(raw_df)
df = mapped_df.copy()

# Sidebar: Show mapped info
if st.sidebar.checkbox("Show column mapping"):
    st.sidebar.json(detected_map)

# Sidebar: Raw preview
if st.sidebar.checkbox("Show raw data"):
    st.sidebar.dataframe(raw_df.head())


if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

for metric in PRIORITY_METRICS:
    if metric in df.columns:
        df[metric] = pd.to_numeric(df[metric], errors='coerce')

if 'value' in df.columns and 'cost' in df.columns and 'profit' not in df.columns:
    df['profit'] = df['value'].fillna(0) - df['cost'].fillna(0)

if 'value' in df.columns and 'profit' in df.columns:
    df['profit_margin_pct'] = np.where(df['value'] != 0, (df['profit'] / df['value']) * 100, np.nan)



st.sidebar.subheader("🔍 Filters")
work_df = df.copy()

# Date range
if 'date' in work_df.columns and work_df['date'].notna().any():
    min_date, max_date = work_df['date'].min(), work_df['date'].max()
    date_range = st.sidebar.date_input("Date Range", (min_date, max_date))
    if len(date_range) == 2:
        start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        work_df = work_df[(work_df['date'] >= start) & (work_df['date'] <= end)]

# Region filter
if 'region' in work_df.columns:
    regions = work_df['region'].dropna().unique().tolist()
    region_selection = st.sidebar.multiselect("Region", regions, default=regions[:3])
    if region_selection:
        work_df = work_df[work_df['region'].isin(region_selection)]

# Category filter
if 'category' in work_df.columns:
    cats = work_df['category'].dropna().unique().tolist()
    cat_selection = st.sidebar.multiselect("Category", cats, default=cats[:3])
    if cat_selection:
        work_df = work_df[work_df['category'].isin(cat_selection)]

st.write(f"### Filtered records: **{len(work_df):,}**")


# KPI SECTION
st.markdown("## 📈 Key Performance Indicators")

total_sales = work_df['value'].sum() if 'value' in work_df else 0
total_profit = work_df['profit'].sum() if 'profit' in work_df else 0
profit_margin = (total_profit / total_sales * 100) if total_sales else 0

total_orders = work_df['order_id'].nunique() if 'order_id' in work_df else len(work_df)
unique_customers = work_df['customer_id'].nunique() if 'customer_id' in work_df else None

aov = (total_sales / total_orders) if total_orders else None

# MoM Growth
if 'date' in work_df and 'value' in work_df:
    monthly = work_df.set_index('date').resample('M')['value'].sum()
    mom_growth = monthly.pct_change().iloc[-1] * 100 if len(monthly) > 1 else None
else:
    mom_growth = None

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Sales", f"{total_sales:,.2f}")
k2.metric("Total Profit", f"{total_profit:,.2f}")
k3.metric("Profit Margin (%)", f"{profit_margin:.2f}%")
k4.metric("AOV", f"{aov:,.2f}" if aov else "N/A")

k5, k6, k7 = st.columns(3)
k5.metric("Total Orders", f"{total_orders:,}")
k6.metric("Unique Customers", f"{unique_customers:,}" if unique_customers else "N/A")
k7.metric("MoM Growth (%)", f"{mom_growth:.2f}%" if mom_growth else "N/A")

# Quick Insights
st.markdown("### 🔝 Quick Insights")

if "product" in work_df and "value" in work_df:
    st.write("**Top Product:**", work_df.groupby("product")["value"].sum().idxmax())

if "region" in work_df and "value" in work_df:
    st.write("**Top Region:**", work_df.groupby("region")["value"].sum().idxmax())

if "category" in work_df and "value" in work_df:
    st.write("**Top Category:**", work_df.groupby("category")["value"].sum().idxmax())


# VISUALISATION SECTION
st.markdown("## 📊 Visual Insights")

# Product chart
if 'product' in work_df.columns and 'value' in work_df.columns:
    st.subheader("Top Products by Sales")
    prod_sales = work_df.groupby('product')['value'].sum().sort_values(ascending=False).reset_index()
    fig1 = px.bar(prod_sales.head(15), x='product', y='value', title="Top Products")
    st.plotly_chart(fig1, use_container_width=True)

# Region chart
if 'region' in work_df.columns and 'value' in work_df.columns:
    st.subheader("Sales by Region")
    region_sales = work_df.groupby('region')['value'].sum().reset_index()
    fig2 = px.pie(region_sales, names='region', values='value')
    st.plotly_chart(fig2, use_container_width=True)

# Monthly trend
if 'date' in work_df.columns and 'value' in work_df.columns:
    st.subheader("Monthly Sales Trend")
    monthly = work_df.set_index('date').resample('M')['value'].sum().reset_index()
    monthly['month'] = monthly['date'].dt.strftime('%Y-%m')
    fig3 = px.line(monthly, x='month', y='value', markers=True)
    st.plotly_chart(fig3, use_container_width=True)

# Category performance
if 'category' in work_df.columns and 'value' in work_df.columns:
    st.subheader("Category Performance")
    cat = work_df.groupby('category')['value'].sum().reset_index()
    fig4 = px.bar(cat, x='category', y='value')
    st.plotly_chart(fig4, use_container_width=True)


st.success("Dashboard ready! Upload another file or adjust filters to explore more.")
