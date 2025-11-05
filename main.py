import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# ------------------------------
# 🧩 1. Load Data
# ------------------------------
st.set_page_config(page_title="Sales Analyzer Dashboard", layout="wide")
st.title("📊 Sales Analyzer Dashboard")

# Upload file
uploaded_file = st.file_uploader("Upload your sales CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Clean and prepare
    df["Date"] = pd.to_datetime(df["Date"])
    df["Profit Margin (%)"] = (df["Profit"] / df["Total Sales"]) * 100

    st.write("### Data Preview", df.head())

    # ------------------------------
    # 📈 2. KPI Section
    # ------------------------------
    total_sales = df["Total Sales"].sum()
    total_profit = df["Profit"].sum()
    total_units = df["Units Sold"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Sales", f"${total_sales:,.0f}")
    col2.metric("📦 Total Units Sold", f"{total_units:,}")
    col3.metric("💹 Total Profit", f"${total_profit:,.0f}")

    # ------------------------------
    # 🏆 3. Best Selling Product
    # ------------------------------
    st.subheader("🏆 Best Selling Products")
    product_sales = df.groupby("Product")["Total Sales"].sum().sort_values(ascending=False)
    fig1 = px.bar(product_sales, x=product_sales.index, y=product_sales.values,
                  text_auto=True, title="Total Sales by Product", color=product_sales.values)
    st.plotly_chart(fig1, use_container_width=True)

    # ------------------------------
    # 🌍 4. Regional Performance
    # ------------------------------
    st.subheader("🌍 Regional Performance")
    region_sales = df.groupby("Region")["Total Sales"].sum()
    fig2, ax = plt.subplots()
    ax.pie(region_sales, labels=region_sales.index, autopct='%1.1f%%', startangle=90)
    ax.set_title("Sales Distribution by Region")
    st.pyplot(fig2)

    # ------------------------------
    # 💰 5. Profit Margin by Product
    # ------------------------------
    st.subheader("💰 Profit Margin by Product")
    avg_margin = df.groupby("Product")["Profit Margin (%)"].mean()
    fig3 = px.bar(avg_margin, x=avg_margin.index, y=avg_margin.values,
                  text_auto=".2f", color=avg_margin.values,
                  title="Average Profit Margin (%) by Product")
    st.plotly_chart(fig3, use_container_width=True)

    # ------------------------------
    # 📅 6. Sales Trend Over Time
    # ------------------------------
    st.subheader("📅 Sales Trend Over Time")
    daily_sales = df.groupby("Date")["Total Sales"].sum()
    fig4 = px.line(x=daily_sales.index, y=daily_sales.values, markers=True,
                   title="Daily Total Sales Trend", labels={"x": "Date", "y": "Total Sales"})
    st.plotly_chart(fig4, use_container_width=True)

else:
    st.info("👆 Please upload a CSV file to see the analysis.")
# ------------------------------
# ⚙️ Footer Section
# ------------------------------
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #555;
        text-align: center;
        padding: 10px 0;
        font-size: 15px;
        border-top: 1px solid #ddd;
    }
    .footer a {
        text-decoration: none;
        color: #0068c9;
        margin: 0 8px;
    }
    .footer a:hover {
        text-decoration: underline;
        color: #004b9b;
    }
    </style>

    <div class="footer">
        © 2025 <b>RHS Analytics</b> | Developed by Rakia, Hashim & Suhail<br>
        <a href="https://www.linkedin.com/in/hashim-malik-a868102b0/" target="_blank">LinkedIn</a> •
        <a href="https://github.com/Hashimmalik46" target="_blank">GitHub</a> •
    </div>
""", unsafe_allow_html=True)
