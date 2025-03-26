# Standard library imports
import calendar
import datetime
from datetime import timedelta
import io
import streamlit as st

# Data manipulation imports
import pandas as pd
import numpy as np

# Visualization imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events

# ML imports
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

# Add statsmodels imports
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm


# Custom color scheme
COLOR_PALETTE = {
    'primary': '#2E86C1',
    'secondary': '#28B463',
    'accent': '#E74C3C',
    'neutral': '#566573',
    'background': '#F8F9F9'
}

# Set page configuration
st.set_page_config(
    page_title="Pharmacy Analytics Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
  
    div[data-testid="stMetricValue"] > div {
        font-size: 1.9rem !important;
    }
    div[data-testid="stMetricDelta"] > div {
        font-size: 1rem !important;
    }
    
  
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        border-bottom: 1px dotted black;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
  
    .loading_spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #3498db;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }
            
    </style>
""", unsafe_allow_html=True)

# Add loading spinner utility function
def show_loading_spinner():
    st.markdown('<div class="loading_spinner"></div>', unsafe_allow_html=True)

# Enhanced tooltip utility function with question mark icon
def add_tooltip(text, tooltip_text):
    return f"""
    <div style="display: inline-flex; align-items: center; gap: 4px;">
        {text}
        <span class="tooltip" style="display: inline-block;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="#666" style="cursor: help;">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 17h-2v-2h2v2zm2.07-7.75l-.9.92C13.45 12.9 13 13.5 13 15h-2v-.5c0-1.1.45-2.1 1.17-2.83l1.24-1.26c.37-.36.59-.86.59-1.41 0-1.1-.9-2-2-2s-2 .9-2 2H8c0-2.21 1.79-4 4-4s4 1.79 4 4c0 .88-.36 1.68-.93 2.25z"/>
            </svg>
            <span class="tooltiptext" style="font-size: 14px; width: 200px;">{tooltip_text}</span>
        </span>
    </div>
    """

# Display main header with custom styling
st.markdown("<h1 style='text-align: center; color: #2E86C1; padding: 20px;'>Pharmacy Analytics Dashboard</h1>", unsafe_allow_html=True)

# Load Data
@st.cache_data(ttl=3600)
def load_data(start_date=None, end_date=None):
    file_path = "pharmacy.xlsx"
    xls = pd.ExcelFile(file_path)
    
    # Load all sheets
    lists_df = pd.read_excel(xls, sheet_name="lists")
    daily_income_df = pd.read_excel(xls, sheet_name="Daily Income")
    inventory_purchases_df = pd.read_excel(xls, sheet_name="Inventory Purchases") 
    expenses_df = pd.read_excel(xls, sheet_name="Expenses")
    
    # Clean up column names
    lists_df.columns = lists_df.columns.str.strip().str.lower()
    inventory_purchases_df.columns = inventory_purchases_df.columns.str.strip()
    expenses_df.columns = expenses_df.columns.str.strip()
    
    # Data preprocessing - optimized
    # Convert dates and filter in one pass
    dfs = {
        'daily_income': daily_income_df,
        'inventory': inventory_purchases_df,
        'expenses': expenses_df
    }
    
    for name, df in dfs.items():
        # Convert dates
        df["date"] = pd.to_datetime(df["Date"], errors='coerce')
        # Filter by date range if provided
        if start_date and end_date:
            mask = (df["date"] >= pd.to_datetime(start_date)) & \
                   (df["date"] <= pd.to_datetime(end_date))
            dfs[name] = df[mask].copy()
        else:
            dfs[name] = df.copy()
            
        # Fill numeric columns with 0
        numeric_cols = dfs[name].select_dtypes(include=['float64', 'int64']).columns
        dfs[name][numeric_cols] = dfs[name][numeric_cols].fillna(0)
    
    # Calculate derived columns using vectorized operations
    dfs['daily_income']["net_income"] = dfs['daily_income']["Total"] - \
                                      dfs['expenses']["Expense Amount"] - \
                                      dfs['inventory']["Invoice Amount"]
    dfs['daily_income']["deficit"] = dfs['daily_income']["Total"] - \
                                   dfs['daily_income']["Gross Income_sys"]
    
    # Remove rows with null dates
    for name in dfs:
        dfs[name] = dfs[name].dropna(subset=["date"])
    
    return {
        "lists": lists_df,
        "daily_income": daily_income_df,
        "inventory_purchases": inventory_purchases_df,
        "expenses": expenses_df
    }

# Load data incrementally based on selected date range
def load_filtered_data(start_date, end_date):
    # First load just the date columns to determine the chunks needed
    date_ranges = []
    full_start = pd.read_excel("pharmacy.xlsx", sheet_name="Daily Income", usecols=["Date"])["Date"].min()
    full_end = pd.read_excel("pharmacy.xlsx", sheet_name="Daily Income", usecols=["Date"])["Date"].max()
    
    # If the date range is more than 3 months, load in monthly chunks
    if (end_date - start_date).days > 90:
        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + pd.DateOffset(months=1), end_date)
            date_ranges.append((current_start, current_end))
            current_start = current_end + pd.DateOffset(days=1)
    else:
        date_ranges.append((start_date, end_date))
    
    # Load data for each date range and concatenate
    all_data = None
    for range_start, range_end in date_ranges:
        chunk = load_data(range_start, range_end)
        if all_data is None:
            all_data = chunk
        else:
            for key in all_data:
                all_data[key] = pd.concat([all_data[key], chunk[key]])
    
    return all_data

# Report generation function
@st.cache_data(ttl=3600)
def generate_report(data, report_type="financial"):
    """Generate comprehensive reports in different formats"""
    output = io.BytesIO()
    
    if report_type == "financial":
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Financial Overview
            data["daily_income"].to_excel(writer, sheet_name='Revenue')
            data["expenses"].to_excel(writer, sheet_name='Expenses')
            data["inventory_purchases"].to_excel(writer, sheet_name='Inventory')
            
    elif report_type == "analytics":
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Analytics sheets
            daily_summary = data["daily_income"].groupby("date")[["Total", "cash", "visa"]].sum()
            daily_summary.to_excel(writer, sheet_name='Daily Summary')
            
    output.seek(0)
    return output

# Schedule settings model
class ReportSchedule:
    def __init__(self, frequency="weekly", day_of_week="Monday", time="09:00"):
        self.frequency = frequency
        self.day_of_week = day_of_week
        self.time = time
        self.last_run = None
        
    def should_run(self):
        now = datetime.datetime.now()
        
        if self.frequency == "daily":
            if self.last_run and (now - self.last_run).days < 1:
                return False
            return now.strftime("%H:%M") == self.time
            
        elif self.frequency == "weekly":
            if self.last_run and (now - self.last_run).days < 7:
                return False
            return (now.strftime("%A") == self.day_of_week and 
                   now.strftime("%H:%M") == self.time)
            
        return False

# Load data with spinner

try:
    with st.spinner("Loading data..."):
        data = load_data()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()


# ====================== SIDEBAR FILTERS ======================
with st.sidebar:
    st.markdown("### 🔍 Dashboard Filters")
    st.divider()
    
    # Date range filter
    st.markdown("#### 📅 Date Range")
    date_preset = st.selectbox(
        "Quick Select",
        ["Custom", "Last 7 Days", "Last 30 Days", "Last 90 Days", "Year to Date", "All Time"],
        key="date_preset"
    )
    
    if date_preset == "Custom":
        date_range = st.date_input(
            "Select Date Range",
            value=(data["daily_income"]["date"].min().date(), 
                   data["daily_income"]["date"].max().date()),
            min_value=data["daily_income"]["date"].min().date(),
            max_value=data["daily_income"]["date"].max().date()
        )
        start_date, end_date = date_range if len(date_range) == 2 else (
            data["daily_income"]["date"].min().date(),
            data["daily_income"]["date"].max().date()
        )
    else:
        end_date = data["daily_income"]["date"].max().date()
        start_date = {
            "Last 7 Days": end_date - timedelta(days=7),
            "Last 30 Days": end_date - timedelta(days=30),
            "Last 90 Days": end_date - timedelta(days=90),
            "Year to Date": datetime.date(end_date.year, 1, 1),
            "All Time": data["daily_income"]["date"].min().date()
        }[date_preset]

    # Month filter
    st.markdown("#### 📅 Select Month")
    month_preset = st.selectbox(
        "Month",
        ["All", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
        key="month_preset"
    )
    
    if month_preset != "All":
        month_number = list(calendar.month_name).index(month_preset)
        filtered_data = {
            "daily_income": data["daily_income"][
                (data["daily_income"]["date"].dt.month == month_number) & 
                (data["daily_income"]["date"].dt.date >= start_date) & 
                (data["daily_income"]["date"].dt.date <= end_date)
            ].copy(),
            "inventory": data["inventory_purchases"][
                (data["inventory_purchases"]["date"].dt.month == month_number) & 
                (data["inventory_purchases"]["date"].dt.date >= start_date) & 
                (data["inventory_purchases"]["date"].dt.date <= end_date)
            ].copy(),
            "expenses": data["expenses"][
                (data["expenses"]["date"].dt.month == month_number) & 
                (data["expenses"]["date"].dt.date >= start_date) & 
                (data["expenses"]["date"].dt.date <= end_date)
            ].copy()
        }
    else:
        filtered_data = {
            "daily_income": data["daily_income"][
                (data["daily_income"]["date"].dt.date >= start_date) & 
                (data["daily_income"]["date"].dt.date <= end_date)
            ].copy(),
            "inventory": data["inventory_purchases"][
                (data["inventory_purchases"]["date"].dt.date >= start_date) & 
                (data["inventory_purchases"]["date"].dt.date <= end_date)
            ].copy(),
            "expenses": data["expenses"][
                (data["expenses"]["date"].dt.date >= start_date) & 
                (data["expenses"]["date"].dt.date <= end_date)
            ].copy()
        }
    # Additional filters
    st.markdown("#### 📦 Inventory")
    inventory_types = sorted(data["inventory_purchases"]["Inventory Type"].unique().tolist())
    selected_type = st.selectbox("Inventory Type", ["All"] + inventory_types, key="sidebar_inventory_type")
    
    st.markdown("#### 🏢 Companies")
    companies = sorted(data["inventory_purchases"]["Invoice Company"].unique().tolist())
    selected_company = st.selectbox("Company", ["All"] + companies, key="sidebar_company")
    
    st.markdown("#### 💰 Expenses")
    expense_types = sorted(data["expenses"]["Expense Type"].unique().tolist())
    selected_expense = st.selectbox("Expense Type", ["All"] + expense_types, key="sidebar_expense_type")
    
    # Apply additional filters
    if selected_type != "All":
        filtered_data["inventory"] = filtered_data["inventory"][
            filtered_data["inventory"]["Inventory Type"] == selected_type
        ]
    
    if selected_company != "All":
        filtered_data["inventory"] = filtered_data["inventory"][
            filtered_data["inventory"]["Invoice Company"] == selected_company
        ]
    
    if selected_expense != "All":
        filtered_data["expenses"] = filtered_data["expenses"][
            filtered_data["expenses"]["Expense Type"] == selected_expense
        ]

# ====================== MAIN DASHBOARD CONTENT ======================

# Main Dashboard Tabs
tab_overview, tab_revenue, tab_inventory, tab_expenses, tab_analytics, tab_ml, tab_search = st.tabs([
    "📊 Overview",
    "💰 Revenue",
    "📦 Inventory",
    "💸 Expenses",
    "📈 Analytics",
    "🤖 ML & Predictions",
    "🔍 Search & Reports"
])


# Overview Tab
with tab_overview:
    st.markdown("### 📊 Key Performance Indicators")

    # --- KPI Calculations (Robustness) ---
    @st.cache_data(ttl=3600)
    def calculate_kpis(_filtered_data):
        if not _filtered_data["daily_income"].empty:
            total_income = _filtered_data["daily_income"]["Total"].sum()
            avg_daily_revenue = _filtered_data["daily_income"]["Total"].mean()
        else:
            total_income = 0
            avg_daily_revenue = 0

        if not _filtered_data["expenses"].empty:
            total_expenses = _filtered_data["expenses"]["Expense Amount"].sum()
        else:
            total_expenses = 0

        if not _filtered_data["inventory"].empty:
            total_purchases = _filtered_data["inventory"]["Invoice Amount"].sum()
        else:
            total_purchases = 0

        net_profit = total_income - total_expenses - total_purchases
        money_deficit = _filtered_data["daily_income"]["deficit"].sum() if not _filtered_data["daily_income"].empty else 0
        
        return {
            'total_income': total_income,
            'avg_daily_revenue': avg_daily_revenue,
            'total_expenses': total_expenses,
            'total_purchases': total_purchases,
            'net_profit': net_profit,
            'money_deficit': money_deficit
        }

    kpis = calculate_kpis(filtered_data)
    total_income = kpis['total_income']
    avg_daily_revenue = kpis['avg_daily_revenue']
    total_expenses = kpis['total_expenses']
    total_purchases = kpis['total_purchases']
    net_profit = kpis['net_profit']
    money_deficit = kpis['money_deficit']

    # First Row - Main KPIs with robust calculations
    kpi_cols = st.columns(5)
    
    with kpi_cols[0]:
        st.markdown(add_tooltip("Total Revenue", "Sum of all income from sales and services"), unsafe_allow_html=True)
        st.metric("", f"EGP {total_income:,.2f}", 
                  delta=f"{(total_income/total_income*100 if total_income > 0 else 0):.1f}% of Total")
    with kpi_cols[1]:
        st.markdown(add_tooltip("Total Expenses", "Sum of all operational expenses (salaries, utilities, etc.) excluding inventory purchases"), unsafe_allow_html=True)
        st.metric("", f"EGP {total_expenses:,.2f}",
                  delta=f"{(total_expenses/total_income*100 if total_income > 0 else 0):.1f}% of Revenue")
    with kpi_cols[2]:
        st.markdown(add_tooltip("Total Purchases", "Sum of all inventory/drug purchases from suppliers"), unsafe_allow_html=True)
        st.metric("", f"EGP {total_purchases:,.2f}",
                  delta=f"{(total_purchases/total_income*100 if total_income > 0 else 0):.1f}% of Revenue")
    with kpi_cols[3]:
        st.markdown(add_tooltip("Net Profit", "Revenue − (Expenses + Purchases) = Gross profit"), unsafe_allow_html=True)
        st.metric("", f"EGP {net_profit:,.2f}",
                  delta=f"{(net_profit/total_income*100 if total_income > 0 else 0):.1f}% Margin")
    with kpi_cols[4]:
        st.markdown(add_tooltip("Money Deficit", "Difference between actual cash revenue and system recorded revenue. May indicate unrecorded sales."), unsafe_allow_html=True)
        st.metric("", f"EGP {money_deficit:,.2f}",
                  delta=f"{(money_deficit/total_income*100 if total_income > 0 else 0):.1f}% of Revenue")

    st.markdown("---")

# Second Row - Payment Methods
    st.markdown("### 💳 Payment Methods Analysis")
    payment_cols = st.columns(4)
    
    total_cash = filtered_data["daily_income"]["cash"].sum()
    total_visa = filtered_data["daily_income"]["visa"].sum()
    total_due = filtered_data["daily_income"]["due amount"].sum()
    
    with payment_cols[0]:
        st.markdown(add_tooltip("Cash Payments", "Physically collected cash payments - should match till amounts"), unsafe_allow_html=True)
        st.metric("", f"EGP {total_cash:,.2f}",
                 delta=f"{(total_cash/total_income*100):.1f}% of Revenue")
    
    with payment_cols[1]:
        st.markdown(add_tooltip("Visa Payments", "Digital card payments via Visa machines"), unsafe_allow_html=True)
        st.metric("", f"EGP {total_visa:,.2f}",
                 delta=f"{(total_visa/total_income*100):.1f}% of Revenue")
    
    with payment_cols[2]:
        st.markdown(add_tooltip("Due Amounts", "Credit/IOU amounts awaiting collection"), unsafe_allow_html=True)
        st.metric("", f"EGP {total_due:,.2f}",
                 delta=f"{(total_due/total_income*100):.1f}% of Revenue")
    
    with payment_cols[3]:
        st.markdown(add_tooltip("System Income", "Theoretical income based on drug prices in system (should match Revenue)"), unsafe_allow_html=True)
        st.metric("", f"EGP {filtered_data['daily_income']['Gross Income_sys'].sum():,.2f}",
                 delta=f"{(filtered_data['daily_income']['Gross Income_sys'].sum()/total_income*100):.1f}%")


    # Enhanced Data Quality Indicators
    st.markdown("### 🛠️ Data Quality Indicators")
    quality_cols = st.columns(4)
    
    with quality_cols[0]:
        # Data Completeness
        income_missing = data["daily_income"].isnull().sum().sum()
        income_total = data["daily_income"].size
        income_completeness = (1 - (income_missing/income_total)) * 100
        
        st.markdown(add_tooltip("📊 Data Completeness", "Percentage of non-empty values in the dataset"), unsafe_allow_html=True)
        st.metric("", 
                 f"{income_completeness:.1f}%",
                 f"Missing: {income_missing} values")
    
    with quality_cols[1]:
        # Data Consistency
        date_range = (data["daily_income"]["date"].max() - data["daily_income"]["date"].min()).days
        expected_days = (datetime.datetime.now().date() - data["daily_income"]["date"].min().date()).days
        consistency = (date_range / expected_days * 100) if expected_days > 0 else 100
        
        st.markdown(add_tooltip("🔄 Data Consistency", "How complete the date coverage is for the selected period"), unsafe_allow_html=True)
        st.metric("", 
                 f"{consistency:.1f}%",
                 f"Coverage: {date_range} of {expected_days} days")
    
    with quality_cols[2]:
        # Data Validity
        invalid_dates = data["daily_income"]["date"].isnull().sum()
        negative_values = (data["daily_income"]["Total"] < 0).sum()
        
        st.markdown(add_tooltip("✅ Data Validity", "Check for invalid dates or impossible negative values"), unsafe_allow_html=True)
        st.metric("",
                f"{(1 - (invalid_dates + negative_values)/len(data['daily_income'])) * 100:.1f}%",
                f"Invalid: {invalid_dates} dates, {negative_values} negative values")
    
    with quality_cols[3]:
        # Data Freshness
        days_since_update = (datetime.datetime.now() - data["daily_income"]["date"].max()).days
        freshness_status = "Fresh" if days_since_update <= 1 else "Stale" if days_since_update <= 7 else "Outdated"
        
        st.markdown(add_tooltip("⏱️ Data Freshness", "How recently the data was updated"), unsafe_allow_html=True)
        st.metric("",
                freshness_status,
                f"Updated {days_since_update} days ago")

    st.markdown("---")

    # --- Financial Health Indicators ---
    st.markdown("### 📈 Financial Health Indicators")
    health_cols = st.columns(3)

    with health_cols[0]:
        profit_margin = (net_profit / total_income * 100) if total_income > 0 else 0
        fig_margin = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=profit_margin,
            title={"text": "Profit Margin (%)", "font": {"size": 20}},
            delta={'reference': 30, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": "black"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 15], "color": "red"},
                    {"range": [15, 30], "color": "yellow"},
                    {"range": [30, 100], "color": "green"},
                ],
            }
        ))
        st.plotly_chart(fig_margin, use_container_width=True)

    with health_cols[1]:
        expense_ratio = (total_expenses / total_income * 100) if total_income > 0 else 0
        fig_expense_ratio = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=expense_ratio,
            title={"text": "Expense Ratio (%)", 'font': {'size': 20}},
            delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": "black"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 25], 'color': "green"},
                    {'range': [25, 40], 'color': "yellow"},
                    {'range': [40, 100], 'color': "red"}
                ],
            }
        ))
        st.plotly_chart(fig_expense_ratio, use_container_width=True)

    with health_cols[2]:
        purchase_to_income_ratio = (total_purchases / total_income * 100) if total_income > 0 else 0
        fig_ratio = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=purchase_to_income_ratio,
            title={'text': "Purchases to Income Ratio (%)", 'font': {'size': 20}},
            delta={'reference': 60, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "black"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': 'yellow'},
                    {'range': [40, 60], 'color': 'green'},
                    {'range': [60, 100], 'color': 'red'}]
            }))
        st.plotly_chart(fig_ratio, use_container_width=True)

      # --- Critical KPI Notice (New) ---
    # Initialize notices for each category
    notices = {
        "profit": {"message": "", "color": ""},
        "expense": {"message": "", "color": ""}, 
        "inventory": {"message": "", "color": ""}
    }

    # Profit Margin Checks
    if profit_margin < 20:
        notices["profit"] = {
            "message": "Profit Margin is critically low! Focus on increasing revenue or reducing costs.",
            "color": "#EF5A6F"  # Light red
        }
    elif profit_margin < 50:
        notices["profit"] = {
            "message": "Profit Margin needs improvement. Consider strategies to increase profitability.",
            "color": "#FFB22C"  # Light orange
        }
    else:
        notices["profit"] = {
            "message": "Profit Margin is healthy.",
            "color": "#219C90"  # Light green
        }

    # Expense Ratio Checks
    if expense_ratio > 50:
        notices["expense"] = {
            "message": "Expense Ratio is very high! Immediate action is needed to control expenses.",
            "color": "#EF5A6F"
        }
    elif expense_ratio > 30:
        notices["expense"] = {
            "message": "Expense Ratio is above target. Review and optimize expenses.",
            "color": "#FFB22C"
        }
    else:
        notices["expense"] = {
            "message": "Expense Ratio is within acceptable range.",
            "color": "#219C90"
        }

    # Inventory to Income Ratpurchase_to_income_ratio
    if purchase_to_income_ratio > 60:
        notices["inventory"] = {
            "message": "Inventory purchases are too high compared to income! Review purchasing strategy.",
            "color": "#EF5A6F"
        }
    elif purchase_to_income_ratio > 40:
        notices["inventory"] = {
            "message": "Inventory to income ratio is healthy.",
            "color": "#219C90"
        }
    else:
        notices["inventory"] = {
            "message": "Inventory to income ratio is elevated. Consider optimizing purchases.",
            "color": "#FFB22C"
        }


    # Display notices in three columns
    cols = st.columns(3)
    for idx, (category, notice) in enumerate(notices.items()):
        with cols[idx]:
            st.markdown(
                f"""
                <div style="
                    border-radius: 10px;
                    text-align: center;
                    color: white;
                    font-size: 20px;
                    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
                    background-color: {notice['color']};
                    height: 100px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;">
                    <div style="
                        font-size: 18px;
                        margin-bottom: 8px;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                        font-weight: bold;">
                        {category.title()}
                    </div>
                    <div style="font-size: 16px;">
                        {notice['message']}
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )

    st.markdown("---")

   
# --- Revenue vs. Expenses Chart (Enhanced) ---
    st.markdown("### 💰 Revenue vs. Expenses Analysis")
    period = st.selectbox("Select Time Period", ["Daily", "Weekly", "Monthly"], key="rev_exp_period")
    
    if not filtered_data["daily_income"].empty and not filtered_data["expenses"].empty:
        # Convert date to period
        if period == "Weekly":
            filtered_data["daily_income"]["period"] = pd.to_datetime(filtered_data["daily_income"]["date"]).dt.to_period('W')
            filtered_data["expenses"]["period"] = pd.to_datetime(filtered_data["expenses"]["date"]).dt.to_period('W')
            filtered_data["inventory"]["period"] = pd.to_datetime(filtered_data["inventory"]["date"]).dt.to_period('W')
        elif period == "Monthly":
            filtered_data["daily_income"]["period"] = pd.to_datetime(filtered_data["daily_income"]["date"]).dt.to_period('M')
            filtered_data["expenses"]["period"] = pd.to_datetime(filtered_data["expenses"]["date"]).dt.to_period('M')
            filtered_data["inventory"]["period"] = pd.to_datetime(filtered_data["inventory"]["date"]).dt.to_period('M')
        else:  # Daily
            filtered_data["daily_income"]["period"] = filtered_data["daily_income"]["date"]
            filtered_data["expenses"]["period"] = filtered_data["expenses"]["date"]
            filtered_data["inventory"]["period"] = filtered_data["inventory"]["date"]
            
        daily_data = filtered_data["daily_income"].groupby("period")["Total"].sum().reset_index()
        daily_data = daily_data.merge(
            filtered_data["expenses"].groupby("period")["Expense Amount"].sum().reset_index(), 
            on="period", 
            how="left"
        )
        daily_data = daily_data.merge(
            filtered_data["inventory"].groupby("period")["Invoice Amount"].sum().reset_index(), 
            on="period", 
            how="left"
        )
        daily_data["Expense Amount"] = daily_data["Expense Amount"].fillna(0)
        daily_data["Invoice Amount"] = daily_data["Invoice Amount"].fillna(0)
        daily_data["net_profit"] = daily_data["Total"] - daily_data["Expense Amount"] - daily_data["Invoice Amount"]

        # Convert period back to datetime for plotting
        if period != "Daily":
            daily_data["period"] = daily_data["period"].dt.to_timestamp()
            
        fig_rev_exp = go.Figure()
        # Add traces
        fig_rev_exp.add_trace(go.Scatter(
            x=daily_data["period"], 
            y=daily_data["Total"], 
            name="Revenue",
            line=dict(color=COLOR_PALETTE["primary"]),
            fill='tozeroy'
        ))
        fig_rev_exp.add_trace(go.Scatter(
            x=daily_data["period"], 
            y=daily_data["Expense Amount"], 
            name="Expenses",
            line=dict(color=COLOR_PALETTE["accent"]),
            fill='tozeroy'
        ))
        fig_rev_exp.add_trace(go.Scatter(
            x=daily_data["period"], 
            y=daily_data["Invoice Amount"], 
            name="Purchases",
            line=dict(color=COLOR_PALETTE["neutral"]),
            fill='tozeroy'
        ))
        fig_rev_exp.add_trace(go.Scatter(
            x=daily_data["period"], 
            y=daily_data["net_profit"], 
            name="Net Profit",
            line=dict(color=COLOR_PALETTE["secondary"], dash='dash')
        ))

        fig_rev_exp.update_layout(
            title=f"{period} Revenue, Expenses, Purchases, and Profit Analysis",
            xaxis_title="Date",
            yaxis_title="Amount (EGP)",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig_rev_exp, use_container_width=True)
    else:
        st.warning("Insufficient data to display Revenue vs Expenses analysis.")
        
    st.markdown("---")

    # --- Moving Averages Analysis ---
    st.markdown("### 📈 Moving Averages Analysis")
    # Reuse the same period selector
    ma_period = st.selectbox("Select Time Period", ["Daily", "Weekly", "Monthly"], key="ma_period")
    
    if not filtered_data["daily_income"].empty and not filtered_data["expenses"].empty:
        # Convert date to period (same logic as revenue vs expenses chart)
        if ma_period == "Weekly":
            filtered_data["daily_income"]["period"] = pd.to_datetime(filtered_data["daily_income"]["date"]).dt.to_period('W')
            filtered_data["expenses"]["period"] = pd.to_datetime(filtered_data["expenses"]["date"]).dt.to_period('W')
            filtered_data["inventory"]["period"] = pd.to_datetime(filtered_data["inventory"]["date"]).dt.to_period('W')
        elif ma_period == "Monthly":
            filtered_data["daily_income"]["period"] = pd.to_datetime(filtered_data["daily_income"]["date"]).dt.to_period('M')
            filtered_data["expenses"]["period"] = pd.to_datetime(filtered_data["expenses"]["date"]).dt.to_period('M')
            filtered_data["inventory"]["period"] = pd.to_datetime(filtered_data["inventory"]["date"]).dt.to_period('M')
        else:  # Daily
            filtered_data["daily_income"]["period"] = filtered_data["daily_income"]["date"]
            filtered_data["expenses"]["period"] = filtered_data["expenses"]["date"]
            filtered_data["inventory"]["period"] = filtered_data["inventory"]["date"]
            
        # Prepare data with same grouping as revenue vs expenses
        ma_data = filtered_data["daily_income"].groupby("period")["Total"].sum().reset_index()
        ma_data = ma_data.merge(
            filtered_data["expenses"].groupby("period")["Expense Amount"].sum().reset_index(), 
            on="period", 
            how="left"
        )
        ma_data = ma_data.merge(
            filtered_data["inventory"].groupby("period")["Invoice Amount"].sum().reset_index(), 
            on="period", 
            how="left"
        )
        ma_data["Expense Amount"] = ma_data["Expense Amount"].fillna(0)
        ma_data["Invoice Amount"] = ma_data["Invoice Amount"].fillna(0)
        ma_data["net_profit"] = ma_data["Total"] - ma_data["Expense Amount"] - ma_data["Invoice Amount"]

        # Convert period back to datetime for plotting
        if ma_period != "Daily":
            ma_data["period"] = ma_data["period"].dt.to_timestamp()

        # Calculate moving averages based on period
        window_size = 7 if ma_period == "Daily" else 4 if ma_period == "Weekly" else 3
        ma_data['revenue_ma'] = ma_data['Total'].rolling(window=window_size, min_periods=1).mean()
        ma_data['expenses_ma'] = ma_data['Expense Amount'].rolling(window=window_size, min_periods=1).mean()
        ma_data['purchases_ma'] = ma_data['Invoice Amount'].rolling(window=window_size, min_periods=1).mean()
        ma_data['profit_ma'] = ma_data['net_profit'].rolling(window=window_size, min_periods=1).mean()

        # Create moving averages chart
        fig_ma = go.Figure()
        # Add moving average traces
        fig_ma.add_trace(go.Scatter(
            x=ma_data["period"], 
            y=ma_data["revenue_ma"], 
            name="Revenue MA",
            line=dict(color=COLOR_PALETTE["primary"], width=3)
        ))
        fig_ma.add_trace(go.Scatter(
            x=ma_data["period"], 
            y=ma_data["expenses_ma"], 
            name="Expenses MA",
            line=dict(color=COLOR_PALETTE["accent"], width=3)
        ))
        fig_ma.add_trace(go.Scatter(
            x=ma_data["period"], 
            y=ma_data["purchases_ma"], 
            name="Purchases MA",
            line=dict(color=COLOR_PALETTE["neutral"], width=3)
        ))
        fig_ma.add_trace(go.Scatter(
            x=ma_data["period"], 
            y=ma_data["profit_ma"], 
            name="Profit MA",
            line=dict(color=COLOR_PALETTE["secondary"], width=3, dash='dash')
        ))

        fig_ma.update_layout(
            title=f"{ma_period} Moving Averages (Window: {window_size})",
            xaxis_title="Date",
            yaxis_title="Amount (EGP)",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig_ma, use_container_width=True)
    else:
        st.warning("Insufficient data to display Moving Averages analysis.")
        
    st.markdown("---")

    # Third Row - Daily Trends
    st.markdown("### 📈 Daily Performance")
    daily_cols = st.columns(2)
    
    with daily_cols[0]:
        # Daily Revenue vs System Revenue
        daily_comparison = filtered_data["daily_income"].groupby("Date").agg({
            "Total": "sum",
            "Gross Income_sys": "sum"
        }).reset_index()
        
        fig_daily_comp = go.Figure()
        fig_daily_comp.add_trace(go.Scatter(
            x=daily_comparison["Date"],
            y=daily_comparison["Total"],
            name="Actual Revenue",
            line=dict(color=COLOR_PALETTE["primary"])
        ))
        fig_daily_comp.add_trace(go.Scatter(
            x=daily_comparison["Date"],
            y=daily_comparison["Gross Income_sys"],
            name="System Revenue",
            line=dict(color=COLOR_PALETTE["secondary"])
        ))
        fig_daily_comp.update_layout(
            title="Daily Revenue vs System Revenue",
            xaxis_title="Date",
            yaxis_title="Amount (EGP)",
        )
        st.plotly_chart(fig_daily_comp, use_container_width=True)
    
    with daily_cols[1]:
        # Daily Payment Methods
        daily_payments = filtered_data["daily_income"].groupby("Date").agg({
            "cash": "sum",
            "visa": "sum",
            "due amount": "sum"
        }).reset_index()
        
        fig_payments = go.Figure()
        for payment_type in ["cash", "visa", "due amount"]:
            fig_payments.add_trace(go.Bar(
                x=daily_payments["Date"],
                y=daily_payments[payment_type],
                name=payment_type.title()
            ))
        fig_payments.update_layout(
            title="Daily Payment Methods Distribution",
            xaxis_title="Date",
            yaxis_title="Amount (EGP)",
            barmode="stack"
        )
        st.plotly_chart(fig_payments, use_container_width=True)

    st.markdown("---")

    # Fourth Row - Expense Analysis
    st.markdown("### 💸 Expense Breakdown")
    expense_cols = st.columns(2)
    
    with expense_cols[0]:
        # Expense Types Distribution
        expense_by_type = filtered_data["expenses"].groupby("Expense Type")["Expense Amount"].sum()
        fig_expense = go.Figure(data=[go.Pie(
            labels=expense_by_type.index,
            values=expense_by_type.values,
            hole=0.4
        )])
        fig_expense.update_layout(
            title="Expense Distribution by Type",
        )
        st.plotly_chart(fig_expense, use_container_width=True)
    
    with expense_cols[1]:
        # Daily Expenses Trend
        daily_expenses = filtered_data["expenses"].groupby("Date")["Expense Amount"].sum().reset_index()
        fig_exp_trend = go.Figure()
        fig_exp_trend.add_trace(go.Scatter(
            x=daily_expenses["Date"],
            y=daily_expenses["Expense Amount"],
            mode="lines+markers",
            line=dict(color=COLOR_PALETTE["accent"])
        ))
        fig_exp_trend.update_layout(
            title="Daily Expenses Trend",
            xaxis_title="Date",
            yaxis_title="Amount (EGP)",
        )
        st.plotly_chart(fig_exp_trend, use_container_width=True)

    st.markdown("---")

    # Fifth Row - Inventory Analysis
    st.markdown("### 📦 Inventory Insights")
    inventory_cols = st.columns(2)
    
    with inventory_cols[0]:
        # Inventory by Company
        inv_by_company = filtered_data["inventory"].groupby("Invoice Company")["Invoice Amount"].sum().sort_values(ascending=True)
        fig_inv_company = go.Figure(data=[go.Bar(
            y=inv_by_company.index,
            x=inv_by_company.values,
            orientation="h",
            marker_color=COLOR_PALETTE["primary"]
        )])
        fig_inv_company.update_layout(
            title="Purchases by Company",
            xaxis_title="Amount (EGP)",
        )
        st.plotly_chart(fig_inv_company, use_container_width=True)
    
    with inventory_cols[1]:
        # Inventory Types
        inv_by_type = filtered_data["inventory"].groupby("Inventory Type")["Invoice Amount"].sum()
        fig_inv_type = go.Figure(data=[go.Pie(
            labels=inv_by_type.index,
            values=inv_by_type.values,
            hole=0.4
        )])
        fig_inv_type.update_layout(
            title="Distribution by Inventory Type",
        )
        st.plotly_chart(fig_inv_type, use_container_width=True)

    st.markdown("#### 📥 Export Complete Dashboard Report")
    if st.button("Generate Complete Dashboard Report"):
        with st.spinner("Generating comprehensive report..."):
            show_loading_spinner()
            # Create Excel writer
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Financial Overview
                financial_summary = pd.DataFrame({
                    'Metric': ['Total Revenue', 'Total Expenses', 'Total Purchases', 'Net Profit', 'Money Deficit'],
                    'Amount': [total_income, total_expenses, total_purchases, net_profit, money_deficit],
                    'Percentage of Revenue': [100, 
                                           (total_expenses/total_income*100) if total_income > 0 else 0,
                                           (total_purchases/total_income*100) if total_income > 0 else 0,
                                           (net_profit/total_income*100) if total_income > 0 else 0,
                                           (money_deficit/total_income*100) if total_income > 0 else 0]
                })
                financial_summary.to_excel(writer, sheet_name='Financial Overview', index=False)
                total_revenue = filtered_data["daily_income"]["Total"].sum()
                cash_percentage = (filtered_data["daily_income"]["cash"].sum() / total_revenue * 100) if total_revenue > 0 else 0
                visa_percentage = (filtered_data["daily_income"]["visa"].sum() / total_revenue * 100) if total_revenue > 0 else 0
                due_percentage = (filtered_data["daily_income"]["due amount"].sum() / total_revenue * 100) if total_revenue > 0 else 0

                # Payment Methods Analysis
                payment_summary = pd.DataFrame({
                    'Payment Type': ['Cash', 'Visa', 'Due Amount'],
                    'Amount': [total_cash, total_visa, total_due],
                    'Percentage': [cash_percentage, visa_percentage, due_percentage]
                })
                payment_summary.to_excel(writer, sheet_name='Payment Analysis', index=False)
                
                # Daily Performance
                daily_performance = filtered_data["daily_income"].groupby("date").agg({
                    'Total': 'sum',
                    'cash': 'sum',
                    'visa': 'sum',
                    'due amount': 'sum',
                    'Gross Income_sys': 'sum',
                    'deficit': 'sum'
                }).round(2)
                daily_performance.to_excel(writer, sheet_name='Daily Performance')
                
                # Expense Analysis
                expense_analysis = filtered_data["expenses"].pivot_table(
                    values='Expense Amount',
                    index='date',
                    columns='Expense Type',
                    aggfunc='sum',
                    fill_value=0
                ).round(2)
                expense_analysis.to_excel(writer, sheet_name='Expense Analysis')
                
                # Inventory Analysis
                inventory_analysis = filtered_data["inventory"].pivot_table(
                    values='Invoice Amount',
                    index='date',
                    columns=['Inventory Type', 'Invoice Company'],
                    aggfunc='sum',
                    fill_value=0
                ).round(2)
                inventory_analysis.to_excel(writer, sheet_name='Inventory Analysis')
                
                # KPI Metrics
                kpi_metrics = pd.DataFrame({
                    'Metric': ['Profit Margin', 'Expense Ratio', 'Purchase to Income Ratio'],
                    'Value': [profit_margin, expense_ratio, purchase_to_income_ratio],
                    'Status': [
                        'Healthy' if profit_margin >= 30 else 'Needs Improvement' if profit_margin >= 15 else 'Critical',
                        'Good' if expense_ratio <= 25 else 'Warning' if expense_ratio <= 40 else 'Critical',
                        'Optimal' if purchase_to_income_ratio <= 40 else 'Warning' if purchase_to_income_ratio <= 60 else 'Critical'
                    ]
                })
                kpi_metrics.to_excel(writer, sheet_name='KPI Metrics', index=False)
                
                # Create a worksheet for charts
                workbook = writer.book
                worksheet = workbook.add_worksheet('Charts')
                
                # Add title formats
                title_format = workbook.add_format({
                    'bold': True,
                    'font_size': 14,
                    'align': 'center',
                    'valign': 'vcenter'
                })
                
                # Add charts
                revenue_chart = workbook.add_chart({'type': 'line'})
                revenue_chart.add_series({
                    'name': 'Revenue',
                    'categories': '=Daily Performance!$A$2:$A$' + str(len(daily_performance) + 1),
                    'values': '=Daily Performance!$B$2:$B$' + str(len(daily_performance) + 1),
                })
                revenue_chart.set_title({'name': 'Revenue Trend'})
                worksheet.insert_chart('A1', revenue_chart)

            # Create download button
            output.seek(0)
            st.download_button(
                label="📥 Download Complete Report",
                data=output,
                file_name=f"pharmacy_complete_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# Revenue Tab
with tab_revenue:
    st.markdown("### 💰 Revenue Analysis")
    
    # Main Revenue Metrics
    st.markdown("#### 📊 Primary Revenue Metrics")
    revenue_kpi_cols = st.columns(4)
    
    with revenue_kpi_cols[0]:
        total_revenue = filtered_data["daily_income"]["Total"].sum()
        st.markdown(add_tooltip("Total Revenue", "Total income from sales and services for the selected period."), unsafe_allow_html=True)
        st.metric("", f"EGP {total_revenue:,.2f}")
        st.markdown(f"**Daily Average:** EGP {filtered_data['daily_income']['Total'].mean():,.2f}")
        st.markdown(f"**Monthly Average:** EGP {total_revenue/((filtered_data['daily_income']['date'].max() - filtered_data['daily_income']['date'].min()).days/30):,.2f}")
    
    with revenue_kpi_cols[1]:
        cash_percentage = (filtered_data["daily_income"]["cash"].sum() / total_revenue * 100) if total_revenue > 0 else 0
        st.markdown(add_tooltip("Cash Revenue %", "Percentage of total revenue received as cash."), unsafe_allow_html=True)
        st.metric("", f"{cash_percentage:.1f}%")
        st.markdown(f"**Cash Total:** EGP {filtered_data['daily_income']['cash'].sum():,.2f}")
        st.markdown(f"**Daily Cash Avg:** EGP {filtered_data['daily_income']['cash'].mean():,.2f}")
    
    with revenue_kpi_cols[2]:
        visa_percentage = (filtered_data["daily_income"]["visa"].sum() / total_revenue * 100) if total_revenue > 0 else 0
        st.markdown(add_tooltip("Visa Revenue %", "Percentage of total revenue received via Visa/card payments."), unsafe_allow_html=True)
        st.metric("", f"{visa_percentage:.1f}%")
        st.markdown(f"**Visa Total:** EGP {filtered_data['daily_income']['visa'].sum():,.2f}")
        st.markdown(f"**Daily Visa Avg:** EGP {filtered_data['daily_income']['visa'].mean():,.2f}")
    
    with revenue_kpi_cols[3]:
        due_percentage = (filtered_data["daily_income"]["due amount"].sum() / total_revenue * 100) if total_revenue > 0 else 0
        st.markdown(add_tooltip("Due Amount %", "Percentage of total revenue recorded as due/credit."), unsafe_allow_html=True)
        st.metric("", f"{due_percentage:.1f}%")
        st.markdown(f"**Due Total:** EGP {filtered_data['daily_income']['due amount'].sum():,.2f}")
        st.markdown(f"**Daily Due Avg:** EGP {filtered_data['daily_income']['due amount'].mean():,.2f}")

    # Revenue Growth Analysis
    st.markdown("#### 📈 Revenue Growth Analysis")
    growth_cols = st.columns(2)
    
    with growth_cols[0]:
        # Month-over-Month Growth
        monthly_revenue = filtered_data["daily_income"].groupby(filtered_data["daily_income"]["date"].dt.strftime('%Y-%m'))["Total"].sum()
        mom_growth = ((monthly_revenue.iloc[-1] - monthly_revenue.iloc[-2]) / monthly_revenue.iloc[-2] * 100) if len(monthly_revenue) >= 2 else 0
        st.markdown(add_tooltip("Month-over-Month Growth", "Percentage change in total revenue compared to the previous month."), unsafe_allow_html=True)
        st.metric("", f"{mom_growth:.1f}%")
        
        # Weekly Growth Trend
        st.markdown(add_tooltip("Weekly Revenue Trend", "Total revenue aggregated by week over the selected period."), unsafe_allow_html=True)
        weekly_revenue = filtered_data["daily_income"].groupby(filtered_data["daily_income"]["date"].dt.strftime('%Y-%W'))["Total"].sum()
        fig_weekly = go.Figure()
        fig_weekly.add_trace(go.Scatter(x=weekly_revenue.index, y=weekly_revenue.values, mode='lines+markers'))
        fig_weekly.update_layout(title="", height=300) # Title provided by markdown tooltip
        st.plotly_chart(fig_weekly, use_container_width=True)
    
    with growth_cols[1]:
        # Revenue Distribution by Day of Week
        st.markdown(add_tooltip("Average Revenue by Day of Week", "Average daily revenue for each day of the week, with error bars indicating standard deviation."), unsafe_allow_html=True)
        dow_revenue = filtered_data["daily_income"].groupby(filtered_data["daily_income"]["date"].dt.day_name())["Total"].agg(["mean", "std"])
        fig_dow = go.Figure()
        fig_dow.add_trace(go.Bar(x=dow_revenue.index, y=dow_revenue["mean"], error_y=dict(type='data', array=dow_revenue["std"])))
        fig_dow.update_layout(title="", height=300) # Title provided by markdown tooltip
        st.plotly_chart(fig_dow, use_container_width=True)

    # Payment Analysis
    st.markdown("#### 💳 Detailed Payment Analysis")
    payment_cols = st.columns(3)
    
    with payment_cols[0]:
        # Payment Method Trends
        st.markdown(add_tooltip("Payment Method Trends", "Daily trends for cash, visa, and due amount revenue over the selected period."), unsafe_allow_html=True)
        payment_trends = filtered_data["daily_income"][["date", "cash", "visa", "due amount"]].melt(id_vars=["date"])
        fig_payment_trends = px.line(payment_trends, x="date", y="value", color="variable", title="")
        fig_payment_trends.update_layout(height=300)
        st.plotly_chart(fig_payment_trends, use_container_width=True)
        
    with payment_cols[1]:
        # Daily Payment Mix
        st.markdown(add_tooltip("Daily Payment Mix Distribution", "Box plot showing the distribution of the daily percentage contribution of each payment method to total revenue."), unsafe_allow_html=True)
        daily_mix = filtered_data["daily_income"][["cash", "visa", "due amount"]].div(filtered_data["daily_income"]["Total"], axis=0)
        fig_mix = go.Figure()
        for col in daily_mix.columns:
            fig_mix.add_trace(go.Box(y=daily_mix[col], name=col))
        fig_mix.update_layout(title="", height=300)
        st.plotly_chart(fig_mix, use_container_width=True)
        
    with payment_cols[2]:
        # Payment Method Correlations
        st.markdown(add_tooltip("Payment Method Correlations", "Heatmap showing the correlation coefficients between daily cash, visa, and due amounts. Values closer to 1 or -1 indicate stronger correlations."), unsafe_allow_html=True)
        payment_corr = filtered_data["daily_income"][["cash", "visa", "due amount"]].corr()
        fig_corr = go.Figure(data=go.Heatmap(z=payment_corr, x=payment_corr.columns, y=payment_corr.index))
        fig_corr.update_layout(title="", height=300)
        st.plotly_chart(fig_corr, use_container_width=True)

    # Revenue Performance Indicators
    st.markdown("#### 🎯 Revenue Performance Indicators")
    perf_cols = st.columns(4)
    
    with perf_cols[0]:
        revenue_volatility = filtered_data["daily_income"]["Total"].std() / filtered_data["daily_income"]["Total"].mean()
        st.markdown(add_tooltip("Revenue Volatility", "Coefficient of variation (Std Dev / Mean) for daily revenue. Higher values indicate more fluctuation."), unsafe_allow_html=True)
        st.metric("", f"{revenue_volatility:.2f}")
        
    with perf_cols[1]:
        revenue_skewness = filtered_data["daily_income"]["Total"].skew()
        st.markdown(add_tooltip("Revenue Skewness", "Measure of the asymmetry of the revenue distribution. Positive skew indicates more high-revenue days, negative skew indicates more low-revenue days."), unsafe_allow_html=True)
        st.metric("", f"{revenue_skewness:.2f}")
        
    with perf_cols[2]:
        peak_revenue = filtered_data["daily_income"]["Total"].max()
        st.markdown(add_tooltip("Peak Revenue", "Highest single-day revenue recorded in the selected period."), unsafe_allow_html=True)
        st.metric("", f"EGP {peak_revenue:,.2f}")
        
    with perf_cols[3]:
        revenue_consistency = (filtered_data["daily_income"]["Total"] > filtered_data["daily_income"]["Total"].mean()).mean() * 100
        st.markdown(add_tooltip("Above Average Days", "Percentage of days where the total revenue exceeded the average daily revenue for the period."), unsafe_allow_html=True)
        st.metric("", f"{revenue_consistency:.1f}%")

    # Revenue Forecasting
    st.markdown("#### 🔮 Revenue Forecasting")
    forecast_cols = st.columns([2, 1])
    
    with forecast_cols[0]:
        # Simple Moving Averages
        ma_periods = [7, 14, 30]
        revenue_df = filtered_data["daily_income"][["date", "Total"]].set_index("date")
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=revenue_df.index, y=revenue_df["Total"], name="Actual"))
        
        st.markdown(add_tooltip("Revenue Moving Averages", "Shows the actual daily revenue along with 7, 14, and 30-day moving averages to smooth out fluctuations and identify trends."), unsafe_allow_html=True)
        for period in ma_periods:
            ma = revenue_df["Total"].rolling(period).mean()
            fig_ma.add_trace(go.Scatter(x=revenue_df.index, y=ma, name=f"{period}-day MA"))
            
        fig_ma.update_layout(title="", height=400)
        st.plotly_chart(fig_ma, use_container_width=True)
        
    with forecast_cols[1]:
        # Revenue Distribution
        st.markdown(add_tooltip("Revenue Distribution", "Histogram showing the frequency of different daily revenue amounts."), unsafe_allow_html=True)
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=filtered_data["daily_income"]["Total"], nbinsx=30))
        fig_dist.update_layout(title="", height=400)
        st.plotly_chart(fig_dist, use_container_width=True)

    # Revenue Segments Analysis
    st.markdown("#### 📊 Revenue Segments")
    
    # Create revenue segments
    filtered_data["daily_income"]["revenue_segment"] = pd.qcut(filtered_data["daily_income"]["Total"], 
                                                             q=4, 
                                                             labels=["Low", "Medium-Low", "Medium-High", "High"])
    
    st.markdown(add_tooltip("Revenue Segment Statistics", "Days are segmented into four quartiles based on total revenue (Low, Medium-Low, Medium-High, High). This table shows statistics for each segment."), unsafe_allow_html=True)
    segment_stats = filtered_data["daily_income"].groupby("revenue_segment").agg({
        "Total": ["count", "mean", "sum"],
        "cash": "sum",
        "visa": "sum",
        "due amount": "sum"
    }).round(2)
    
    st.dataframe(segment_stats, use_container_width=True)

    st.markdown("#### 📊 Revenue Reports")
    report_cols = st.columns(3)
    
    with report_cols[0]:
        st.markdown(add_tooltip("Export Detailed Revenue Report", "Generates an Excel file containing daily revenue details, payment analysis, monthly trends, segment analysis, and key KPIs."), unsafe_allow_html=True)
        if st.button("Export Detailed Revenue Report"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Daily Revenue Details
                daily_revenue = filtered_data["daily_income"].groupby("date").agg({
                    'Total': 'sum',
                    'cash': 'sum',
                    'visa': 'sum',
                    'due amount': 'sum',
                    'Gross Income_sys': 'sum',
                    'net_income': 'sum',
                    'deficit': 'sum'
                }).round(2)
                daily_revenue.to_excel(writer, sheet_name='Daily Revenue')
                
                # Payment Method Analysis
                payment_analysis = pd.DataFrame({
                    'Method': ['Cash', 'Visa', 'Due Amount'],
                    'Total Amount': [total_cash, total_visa, total_due],
                    'Percentage': [cash_percentage, visa_percentage, due_percentage],
                    'Daily Average': [
                        filtered_data['daily_income']['cash'].mean(),
                        filtered_data['daily_income']['visa'].mean(),
                        filtered_data['daily_income']['due amount'].mean()
                    ]
                })
                payment_analysis.to_excel(writer, sheet_name='Payment Analysis', index=False)
                
                # Revenue Growth
                monthly_growth = filtered_data["daily_income"].groupby(
                    filtered_data["daily_income"]["date"].dt.strftime('%Y-%m')
                ).agg({
                    'Total': ['sum', 'mean', 'std'],
                    'cash': 'sum',
                    'visa': 'sum',
                    'due amount': 'sum'
                }).round(2)
                monthly_growth.to_excel(writer, sheet_name='Monthly Analysis')
                
                # Revenue Segments
                segment_analysis = segment_stats.copy()
                segment_analysis.to_excel(writer, sheet_name='Revenue Segments')
                
                # Revenue KPIs
                revenue_kpis = pd.DataFrame({
                    'Metric': ['Total Revenue', 'Average Daily Revenue', 'Revenue Volatility', 'Revenue Skewness',
                             'Peak Revenue', 'Revenue Consistency', 'Month-over-Month Growth'],
                    'Value': [total_revenue, avg_daily_revenue, revenue_volatility, revenue_skewness,
                             peak_revenue, revenue_consistency, mom_growth]
                })
                revenue_kpis.to_excel(writer, sheet_name='Revenue KPIs', index=False)
                
            output.seek(0)
            st.download_button(
                label="📥 Download Revenue Report",
                data=output,
                file_name=f"revenue_detailed_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with report_cols[1]:
        st.markdown(add_tooltip("Generate Payment Analysis Report", "Generates an Excel file focusing on payment methods, including daily mix percentages and summary statistics."), unsafe_allow_html=True)
        if st.button("Generate Payment Analysis Report"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Daily Payment Mix
                payment_mix = filtered_data["daily_income"][["date", "cash", "visa", "due amount", "Total"]].copy()
                payment_mix[["cash_pct", "visa_pct", "due_pct"]] = payment_mix[["cash", "visa", "due amount"]].div(payment_mix["Total"], axis=0) * 100
                payment_mix.round(2).to_excel(writer, sheet_name='Daily Payment Mix')
                
                # Payment Method Statistics
                payment_stats = filtered_data["daily_income"][["cash", "visa", "due amount"]].agg([
                    'count', 'mean', 'std', 'min', 'max'
                ]).round(2)
                payment_stats.to_excel(writer, sheet_name='Payment Statistics')
                
            output.seek(0)
            st.download_button(
                label="📥 Download Payment Analysis",
                data=output,
                file_name=f"payment_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with report_cols[2]:
        st.markdown(add_tooltip("Generate Growth Analysis Report", "Generates an Excel file containing weekly revenue growth trends and analysis by day of the week."), unsafe_allow_html=True)
        if st.button("Generate Growth Analysis Report"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Weekly Growth
                weekly_growth = filtered_data["daily_income"].groupby(
                    filtered_data["daily_income"]["date"].dt.strftime('%Y-%W')
                ).agg({
                    'Total': ['sum', 'mean', 'std'],
                    'cash': 'sum',
                    'visa': 'sum',
                    'due amount': 'sum'
                }).round(2)
                weekly_growth.to_excel(writer, sheet_name='Weekly Growth')
                
                # Day of Week Analysis
                dow_analysis = filtered_data["daily_income"].groupby(
                    filtered_data["daily_income"]["date"].dt.day_name()
                ).agg({
                    'Total': ['count', 'sum', 'mean', 'std'],
                    'cash': ['sum', 'mean'],
                    'visa': ['sum', 'mean'],
                    'due amount': ['sum', 'mean']
                }).round(2)
                dow_analysis.to_excel(writer, sheet_name='Day of Week Analysis')
                
            output.seek(0)
            st.download_button(
                label="📥 Download Growth Analysis",
                data=output,
                file_name=f"growth_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# Inventory Tab
with tab_inventory:
    st.markdown("### 📦 Inventory Management Dashboard")
    
    # First Row - Main KPIs
    st.markdown("#### 📊 Primary Metrics")
    kpi_cols = st.columns(5)
    
    # Calculate main KPIs
    total_purchases = filtered_data["inventory"]["Invoice Amount"].sum()
    total_credit = filtered_data["inventory"]["Credit Limit"].sum()
    avg_invoice = filtered_data["inventory"]["Invoice Amount"].mean()
    num_suppliers = filtered_data["inventory"]["Invoice Company"].nunique()
    inventory_types_count = filtered_data["inventory"]["Inventory Type"].nunique()

    with kpi_cols[0]:
        st.markdown(add_tooltip("Total Purchases", "Sum of all inventory purchases from suppliers"), unsafe_allow_html=True)
        st.metric("", f"EGP {total_purchases:,.2f}")
    with kpi_cols[1]:
        st.markdown(add_tooltip("Total Credit Limit", "Combined credit available from all suppliers"), unsafe_allow_html=True)
        st.metric("", f"EGP {total_credit:,.2f}")
    with kpi_cols[2]:
        st.markdown(add_tooltip("Average Invoice", "Mean amount per inventory invoice"), unsafe_allow_html=True)
        st.metric("", f"EGP {avg_invoice:,.2f}")
    with kpi_cols[3]:
        st.markdown(add_tooltip("Active Suppliers", "Number of unique suppliers with purchases"), unsafe_allow_html=True)
        st.metric("", f"{num_suppliers}")
    with kpi_cols[4]:
        st.markdown(add_tooltip("Inventory Categories", "Number of different inventory types purchased"), unsafe_allow_html=True)
        st.metric("", f"{inventory_types_count}")

    # Second Row - Credit Analysis
    st.markdown("#### 💳 Credit Management")
    credit_cols = st.columns(4)
    
    # Calculate credit metrics
    credit_utilization = (total_purchases / total_credit * 100) if total_credit > 0 else 0
    avg_credit_limit = filtered_data["inventory"]["Credit Limit"].mean()
    max_credit = filtered_data["inventory"]["Credit Limit"].max()
    
    with credit_cols[0]:
        st.markdown(add_tooltip("Credit Utilization", "Percentage of total credit limit used by total purchases (Purchases / Credit Limit)."), unsafe_allow_html=True)
        st.metric("", f"{credit_utilization:.1f}%")
    with credit_cols[1]:
        st.markdown(add_tooltip("Average Credit Limit", "Average credit limit offered per supplier."), unsafe_allow_html=True)
        st.metric("", f"EGP {avg_credit_limit:,.2f}")
    with credit_cols[2]:
        st.markdown(add_tooltip("Maximum Credit Line", "Highest credit limit offered by any single supplier."), unsafe_allow_html=True)
        st.metric("", f"EGP {max_credit:,.2f}")
    with credit_cols[3]:
        st.markdown(add_tooltip("Credit to Purchase Ratio", "Ratio of total available credit to total purchases (Credit Limit / Purchases). Higher values indicate more available credit relative to spending."), unsafe_allow_html=True)
        st.metric("", 
                 f"{(total_credit/total_purchases if total_purchases > 0 else 0):.2f}x")

    # Third Row - Supplier Analysis
    st.markdown("#### 🏢 Supplier Performance")
    supplier_cols = st.columns(2)
    
    with supplier_cols[0]:
        # Top Suppliers by Volume
        supplier_volume = filtered_data["inventory"].groupby("Invoice Company").agg({
            "Invoice Amount": "sum",
            "id": "count"
        }).sort_values("Invoice Amount", ascending=False)
        
        fig_suppliers = go.Figure()
        fig_suppliers.add_trace(go.Bar(
            x=supplier_volume.head(10).index,
            y=supplier_volume.head(10)["Invoice Amount"],
            name="Purchase Volume",
            marker_color=COLOR_PALETTE["primary"]
        ))
        fig_suppliers.add_trace(go.Scatter(
            x=supplier_volume.head(10).index,
            y=supplier_volume.head(10)["id"],
            name="Number of Invoices",
            yaxis="y2",
            line=dict(color=COLOR_PALETTE["accent"])
        ))
        fig_suppliers.update_layout(
            title="Top 10 Suppliers by Purchase Volume",
            yaxis=dict(title="Purchase Amount (EGP)"),
            yaxis2=dict(title="Number of Invoices", overlaying="y", side="right"),
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig_suppliers, use_container_width=True)
    
    with supplier_cols[1]:
        # Credit Limit Distribution
        st.markdown(add_tooltip("Top 10 Companies by Credit Limit", "Bar chart showing the average credit limit for the top 10 suppliers."), unsafe_allow_html=True)
        credit_dist = filtered_data["inventory"].groupby("Invoice Company")["Credit Limit"].mean()
        fig_credit = go.Figure(data=[go.Bar(
            x=credit_dist.sort_values(ascending=False).head(10).index,
            y=credit_dist.sort_values(ascending=False).head(10).values,
            marker_color=COLOR_PALETTE["secondary"]
        )])
        fig_credit.update_layout(
            title="", # Title provided by markdown tooltip
            xaxis_title="Company",
            yaxis_title="Credit Limit (EGP)",
            height=400
        )
        st.plotly_chart(fig_credit, use_container_width=True)

    # Fourth Row - Inventory Type Analysis
    st.markdown("#### 📦 Inventory Categories")
    type_cols = st.columns(2)
    
    with type_cols[0]:
        # Inventory Type Distribution
        type_dist = filtered_data["inventory"].groupby("Inventory Type").agg({
            "Invoice Amount": "sum",
            "id": "count"
        })
        
        fig_types = go.Figure(data=[go.Pie(
            labels=type_dist.index,
            values=type_dist["Invoice Amount"],
            hole=0.4,
            textinfo="label+percent"
        )])
        fig_types.update_layout(
            title="Purchase Distribution by Inventory Type",
            height=400
        )
        st.plotly_chart(fig_types, use_container_width=True)
    
    with type_cols[1]:
        # Type Trends Over Time
        st.markdown(add_tooltip("Monthly Trends by Inventory Type", "Line chart showing the total purchase amount for each inventory type aggregated monthly."), unsafe_allow_html=True)
        type_trends = filtered_data["inventory"].groupby([
            filtered_data["inventory"]["date"].dt.strftime("%Y-%m"),  # Convert to string format instead of Period
            "Inventory Type"
        ])["Invoice Amount"].sum().reset_index()
        type_trends.columns = ["date", "Inventory Type", "Invoice Amount"]  # Rename columns for clarity
        
        fig_trends = px.line(
            type_trends,
            x="date",
            y="Invoice Amount",
            color="Inventory Type",
            title="" # Title provided by markdown tooltip
        )
        fig_trends.update_layout(
            height=400,
            xaxis_title="Month",
            yaxis_title="Amount (EGP)",
        )
        st.plotly_chart(fig_trends, use_container_width=True)

    # Fifth Row - Invoice Analysis
    st.markdown("#### 📝 Invoice Analytics")
    invoice_cols = st.columns(3)
    
    with invoice_cols[0]:
        # Invoice Size Distribution
        fig_invoice_dist = px.histogram(
            filtered_data["inventory"],
            x="Invoice Amount",
            nbins=50,
            title="Invoice Amount Distribution"
        )
        fig_invoice_dist.update_layout(height=300)
        st.plotly_chart(fig_invoice_dist, use_container_width=True)
    
    with invoice_cols[1]:
        # Daily Invoice Count
        st.markdown(add_tooltip("Daily Invoice Count", "Line chart showing the number of inventory purchase invoices recorded each day."), unsafe_allow_html=True)
        daily_invoices = filtered_data["inventory"].groupby("date")["id"].count()
        fig_daily = go.Figure(data=[go.Scatter(
            x=daily_invoices.index,
            y=daily_invoices.values,
            mode='lines+markers',
            line=dict(color=COLOR_PALETTE["primary"])
        )])
        fig_daily.update_layout(
            title="", # Title provided by markdown tooltip
            height=300
        )
        st.plotly_chart(fig_daily, use_container_width=True)
    
    with invoice_cols[2]:
        # Invoice Type Distribution
        st.markdown(add_tooltip("Distribution by Invoice Type", "Pie chart showing the distribution of total purchase amount by invoice type (e.g., Cash, Credit)."), unsafe_allow_html=True)
        invoice_types = filtered_data["inventory"].groupby("Invoice Type")["Invoice Amount"].sum()
        fig_inv_types = go.Figure(data=[go.Pie(
            labels=invoice_types.index,
            values=invoice_types.values,
            hole=0.4
        )])
        fig_inv_types.update_layout(
            title="", # Title provided by markdown tooltip
            height=300
        )
        st.plotly_chart(fig_inv_types, use_container_width=True)

    # Sixth Row - Detailed Analysis
    st.markdown("#### 📊 Detailed Metrics")
    detail_cols = st.columns(4)
    
    # Calculate additional metrics
    avg_daily_purchase = filtered_data["inventory"].groupby("date")["Invoice Amount"].sum().mean()
    purchase_std = filtered_data["inventory"]["Invoice Amount"].std()
    largest_invoice = filtered_data["inventory"]["Invoice Amount"].max()
    invoice_count = len(filtered_data["inventory"])
    
    with detail_cols[0]:
        st.markdown(add_tooltip("Avg Daily Purchase", "Average total amount of inventory purchased per day."), unsafe_allow_html=True)
        st.metric("", f"EGP {avg_daily_purchase:,.2f}")
    with detail_cols[1]:
        st.markdown(add_tooltip("Purchase Std Dev", "Standard deviation of individual invoice amounts, indicating variability in purchase sizes."), unsafe_allow_html=True)
        st.metric("", f"EGP {purchase_std:,.2f}")
    with detail_cols[2]:
        st.markdown(add_tooltip("Largest Invoice", "The amount of the single largest inventory purchase invoice."), unsafe_allow_html=True)
        st.metric("", f"EGP {largest_invoice:,.2f}")
    with detail_cols[3]:
        st.markdown(add_tooltip("Total Invoices", "Total number of inventory purchase invoices recorded in the period."), unsafe_allow_html=True)
        st.metric("", f"{invoice_count:,}")

    # Seventh Row - Data Table
    st.markdown("#### 📋 Detailed Purchase Records")
    
    # Create expandable detailed view
    with st.expander("View Detailed Purchase Records"):
        # Add search functionality
        search_term = st.text_input("Search by Invoice ID or Company")
        
        if search_term:
            filtered_view = filtered_data["inventory"][
                filtered_data["inventory"].astype(str).apply(
                    lambda x: x.str.contains(search_term, case=False)
                ).any(axis=1)
            ]
        else:
            filtered_view = filtered_data["inventory"]
        
        st.dataframe(
            filtered_view.sort_values("date", ascending=False),
            use_container_width=True
        )

# Expenses Tab
with tab_expenses:
    st.markdown("### 💸 Expense Analysis Dashboard")
    
    # Expense KPIs - Enhanced
    st.markdown("#### 📊 Key Expense Metrics")
    expense_kpi_cols = st.columns(5)
    
    with expense_kpi_cols[0]:
        total_expenses = filtered_data["expenses"]["Expense Amount"].sum()
        st.markdown(add_tooltip("Total Expenses", "Sum of all recorded operational expenses for the selected period."), unsafe_allow_html=True)
        st.metric("", f"EGP {total_expenses:,.2f}",
                 delta=f"{(total_expenses/total_income*100 if total_income > 0 else 0):.1f}% of Revenue")
    
    with expense_kpi_cols[1]:
        avg_expense = filtered_data["expenses"]["Expense Amount"].mean()
        st.markdown(add_tooltip("Avg Daily Expense", "Average amount spent on expenses per day."), unsafe_allow_html=True)
        st.metric("", f"EGP {avg_expense:,.2f}")
    
    with expense_kpi_cols[2]:
        expense_types = filtered_data["expenses"]["Expense Type"].nunique()
        st.markdown(add_tooltip("Expense Categories", "Number of unique expense types recorded."), unsafe_allow_html=True)
        st.metric("", f"{expense_types}")
    
    with expense_kpi_cols[3]:
        max_expense = filtered_data["expenses"]["Expense Amount"].max()
        st.markdown(add_tooltip("Largest Expense", "The amount of the single largest expense recorded."), unsafe_allow_html=True)
        st.metric("", f"EGP {max_expense:,.2f}")
    
    with expense_kpi_cols[4]:
        expense_variance = filtered_data["expenses"]["Expense Amount"].std() / avg_expense * 100 if avg_expense > 0 else 0
        st.markdown(add_tooltip("Expense Variance", "Coefficient of variation (Std Dev / Mean) for daily expenses. Higher values indicate more fluctuation."), unsafe_allow_html=True)
        st.metric("", f"{expense_variance:.1f}%")

    # Expense Distribution and Trends
    st.markdown("#### 📈 Expense Trends & Patterns")
    trend_cols = st.columns(2)
    
    with trend_cols[0]:
        # Enhanced Pie Chart with Budget Comparison
        expense_dist = filtered_data["expenses"].groupby("Expense Type")["Expense Amount"].sum().sort_values(ascending=True)
        fig_pie = go.Figure()
        fig_pie.add_trace(go.Pie(
            labels=expense_dist.index,
            values=expense_dist.values,
            hole=0.4,
            textinfo='label+percent+value',
            textposition='outside',
            marker=dict(colors=px.colors.qualitative.Set3)
        ))
        fig_pie.update_layout(
            title="Expense Distribution by Category",
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with trend_cols[1]:
        # Cumulative Expense Trend
        st.markdown(add_tooltip("Cumulative Expenses Over Time", "Area chart showing the running total of expenses over the selected period."), unsafe_allow_html=True)
        daily_expenses = filtered_data["expenses"].groupby("date")["Expense Amount"].sum().cumsum().reset_index()
        fig_cumulative = px.area(
            daily_expenses,
            x="date",
            y="Expense Amount",
            title="" # Title provided by markdown tooltip
        )
        fig_cumulative.update_layout(height=400)
        st.plotly_chart(fig_cumulative, use_container_width=True)

    # Expense Forecasting
    st.markdown("#### 🔮 Expense Forecasting")
    forecast_cols = st.columns([2, 1])
    
    with forecast_cols[0]:
        # Simple moving average forecast
        expense_series = filtered_data["expenses"].groupby("date")["Expense Amount"].sum()
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=expense_series.index,
            y=expense_series,
            name="Actual Expenses",
            line=dict(color=COLOR_PALETTE['primary'])
        ))
        
        st.markdown(add_tooltip("Expense Trends with Moving Averages", "Line chart showing actual daily expenses along with 7-day and 30-day moving averages."), unsafe_allow_html=True)
        # Add 7-day moving average
        ma7 = expense_series.rolling(7).mean()
        fig_forecast.add_trace(go.Scatter(
            x=ma7.index,
            y=ma7,
            name="7-Day Moving Avg",
            line=dict(color=COLOR_PALETTE['secondary'], dash='dash')
        ))
        
        # Add 30-day moving average
        ma30 = expense_series.rolling(30).mean()
        fig_forecast.add_trace(go.Scatter(
            x=ma30.index,
            y=ma30,
            name="30-Day Moving Avg",
            line=dict(color=COLOR_PALETTE['accent'], dash='dot')
        ))
        
        fig_forecast.update_layout(
            title="", # Title provided by markdown tooltip
            height=400
        )
        st.plotly_chart(fig_forecast, use_container_width=True)
    
    with forecast_cols[1]:
        st.markdown(add_tooltip("Forecast Parameters", "Adjust the forecast period and generate a simple linear regression forecast."), unsafe_allow_html=True)
        st.markdown("##### Forecast Parameters")
        forecast_days = st.slider("Forecast Period (days)", 7, 90, 30)
        
        # Simple forecast calculation
        if st.button("Generate Forecast"):
            with st.spinner("Calculating forecast..."):
                # Simple linear regression forecast
                x = np.arange(len(expense_series)).reshape(-1, 1)
                y = expense_series.values
                model = LinearRegression().fit(x, y)
                future_x = np.arange(len(expense_series), len(expense_series)+forecast_days).reshape(-1, 1)
                future_y = model.predict(future_x)
                
                # Create forecast dataframe
                last_date = expense_series.index[-1]
                future_dates = pd.date_range(start=last_date, periods=forecast_days+1)[1:]
                forecast_df = pd.DataFrame({
                    'date': future_dates,
                    'Expense Amount': future_y
                })
                
                # Display forecast summary
                forecast_total = forecast_df["Expense Amount"].sum()
                forecast_avg = forecast_df["Expense Amount"].mean()
                
                st.markdown(add_tooltip("Forecast Total", "Total predicted expenses for the forecast period."), unsafe_allow_html=True)
                st.metric("", f"EGP {forecast_total:,.2f}")
                st.markdown(add_tooltip("Daily Forecast Avg", "Average daily predicted expense for the forecast period."), unsafe_allow_html=True)
                st.metric("", f"EGP {forecast_avg:,.2f}")
                
                # Add forecast to plot
                fig_forecast.add_trace(go.Scatter(
                    x=forecast_df['date'],
                    y=forecast_df['Expense Amount'],
                    name="Forecast",
                    line=dict(color='red', dash='dot')
                ))
                st.plotly_chart(fig_forecast, use_container_width=True)

    # Expense Category Deep Dive
    st.markdown("#### 🔍 Category Analysis")
    category = st.selectbox(
        "Select Expense Category to Analyze",
        options=filtered_data["expenses"]["Expense Type"].unique()
    )
    
    if category:
        category_data = filtered_data["expenses"][filtered_data["expenses"]["Expense Type"] == category]
        
        cat_cols = st.columns(2)
        with cat_cols[0]:
            # Monthly trend for selected category
            monthly_trend = category_data.groupby(
                category_data["date"].dt.strftime("%Y-%m")
            )["Expense Amount"].sum().reset_index()
            
            st.markdown(add_tooltip(f"Monthly Trend: {category}", f"Bar chart showing the total monthly expense for the '{category}' category."), unsafe_allow_html=True)
            fig_trend = px.bar(
                monthly_trend,
                x="date",
                y="Expense Amount",
                title="" # Title provided by markdown tooltip
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with cat_cols[1]:
            # Statistical distribution
            st.markdown(add_tooltip(f"Distribution: {category}", f"Box plot showing the distribution of daily expense amounts for the '{category}' category."), unsafe_allow_html=True)
            fig_dist = px.box(
                category_data,
                y="Expense Amount",
                title="" # Title provided by markdown tooltip
            )
            st.plotly_chart(fig_dist, use_container_width=True)
    
    # Enhanced Export Options
    st.markdown("#### 📤 Export Expense Reports")
    export_cols = st.columns(3)
    
    with export_cols[0]:
        st.markdown(add_tooltip("Export Expense Summary", "Generates an Excel file summarizing expenses by category (sum, mean, count, std dev)."), unsafe_allow_html=True)
        if st.button("Export Expense Summary"):
            summary = filtered_data["expenses"].groupby("Expense Type").agg({
                "Expense Amount": ["sum", "mean", "count", "std"]
            }).round(2)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                summary.to_excel(writer, sheet_name="Expense Summary")
                
            output.seek(0)
            st.download_button(
                label="Download Summary",
                data=output,
                file_name="expense_summary.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with export_cols[1]:
        st.markdown(add_tooltip("Export Detailed Expenses", "Generates an Excel file containing all individual expense records for the selected period."), unsafe_allow_html=True)
        if st.button("Export Detailed Expenses"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                filtered_data["expenses"].to_excel(writer, sheet_name="Detailed Expenses", index=False)
                
            output.seek(0)
            st.download_button(
                label="Download Details",
                data=output,
                file_name="expense_details.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with export_cols[2]:
        st.markdown(add_tooltip("Export Forecast Data", "Generates an Excel file containing the forecasted expense data (if generated)."), unsafe_allow_html=True)
        if st.button("Export Forecast Data"):
            if 'forecast_df' in locals():
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    forecast_df.to_excel(writer, sheet_name="Forecast", index=False)
                    
                output.seek(0)
                st.download_button(
                    label="Download Forecast",
                    data=output,
                    file_name="expense_forecast.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("Please generate a forecast first")

# Analytics Tab
with tab_analytics:
    st.markdown("### 📈 Advanced Analytics")
    
    # Analytics KPIs
    st.markdown("#### 📊 Analytics KPIs")
    analytics_kpi_cols = st.columns(4)
    
    with analytics_kpi_cols[0]:
        profit_margin = (net_profit / total_income * 100) if total_income > 0 else 0
        st.markdown(add_tooltip("Profit Margin", "Net Profit as a percentage of Total Revenue. Indicates overall profitability."), unsafe_allow_html=True)
        st.metric("", f"{profit_margin:.1f}%")
    
    with analytics_kpi_cols[1]:
        inventory_turnover = total_income / total_purchases if total_purchases > 0 else 0 # Simplified: Revenue / Purchases
        st.markdown(add_tooltip("Inventory Turnover", "Ratio indicating how quickly inventory is sold (approximated as Revenue / Purchases). Higher is generally better."), unsafe_allow_html=True)
        st.metric("", f"{inventory_turnover:.2f}x")
    
    with analytics_kpi_cols[2]:
        avg_transaction = filtered_data["daily_income"]["Total"].mean()
        st.markdown(add_tooltip("Avg Transaction", "Average revenue generated per day."), unsafe_allow_html=True)
        st.metric("", f"EGP {avg_transaction:,.2f}")
    
    with analytics_kpi_cols[3]:
        data_points = len(filtered_data["daily_income"]) + len(filtered_data["inventory"]) + len(filtered_data["expenses"])
        st.markdown(add_tooltip("Total Data Points", "Total number of records across income, inventory, and expense datasets for the selected period."), unsafe_allow_html=True)
        st.metric("", f"{data_points:,}")
    
    # Anomaly Detection
    st.markdown("#### 🚨 Anomaly Detection")
    
    # Enhanced anomaly detection using Z-scores and IQR methods
    if not filtered_data["daily_income"].empty:
        # Revenue anomalies using multiple methods
        revenue_mean = filtered_data["daily_income"]["Total"].mean()
        revenue_std = filtered_data["daily_income"]["Total"].std()
        
        # Z-score method
        filtered_data["daily_income"]["revenue_zscore"] = (
            (filtered_data["daily_income"]["Total"] - revenue_mean) / revenue_std
        )
        
        # IQR method
        q1 = filtered_data["daily_income"]["Total"].quantile(0.25)
        q3 = filtered_data["daily_income"]["Total"].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        
        revenue_anomalies = filtered_data["daily_income"][
            (filtered_data["daily_income"]["revenue_zscore"].abs() > 2) |  # Z-score threshold
            (filtered_data["daily_income"]["Total"] < lower_bound) |      # IQR lower
            (filtered_data["daily_income"]["Total"] > upper_bound)        # IQR upper
        ].copy()
        
        revenue_anomalies["anomaly_type"] = np.where(
            revenue_anomalies["revenue_zscore"].abs() > 2, 
            "Z-score", 
            "IQR"
        )
        
    if not filtered_data["expenses"].empty:
        # Expense anomalies using same dual method
        expense_mean = filtered_data["expenses"]["Expense Amount"].mean()
        expense_std = filtered_data["expenses"]["Expense Amount"].std()
        
        # Z-score method
        filtered_data["expenses"]["expense_zscore"] = (
            (filtered_data["expenses"]["Expense Amount"] - expense_mean) / expense_std
        )
        
        # IQR method
        q1 = filtered_data["expenses"]["Expense Amount"].quantile(0.25)
        q3 = filtered_data["expenses"]["Expense Amount"].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        
        expense_anomalies = filtered_data["expenses"][
            (filtered_data["expenses"]["expense_zscore"].abs() > 2) |
            (filtered_data["expenses"]["Expense Amount"] < lower_bound) |
            (filtered_data["expenses"]["Expense Amount"] > upper_bound)
        ].copy()
        
        expense_anomalies["anomaly_type"] = np.where(
            expense_anomalies["expense_zscore"].abs() > 2, 
            "Z-score", 
            "IQR"
        )
        
    if not filtered_data["inventory"].empty:
        # Purchase anomalies using same dual method
        purchase_mean = filtered_data["inventory"]["Invoice Amount"].mean()
        purchase_std = filtered_data["inventory"]["Invoice Amount"].std()
        
        # Z-score method
        filtered_data["inventory"]["purchase_zscore"] = (
            (filtered_data["inventory"]["Invoice Amount"] - purchase_mean) / purchase_std
        )
        
        # IQR method
        q1 = filtered_data["inventory"]["Invoice Amount"].quantile(0.25)
        q3 = filtered_data["inventory"]["Invoice Amount"].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        
        purchase_anomalies = filtered_data["inventory"][
            (filtered_data["inventory"]["Invoice Amount"] > purchase_mean + 2*purchase_std) |
            (filtered_data["inventory"]["Invoice Amount"] < purchase_mean - 2*purchase_std) |
            (filtered_data["inventory"]["Invoice Amount"] < lower_bound) |
            (filtered_data["inventory"]["Invoice Amount"] > upper_bound)
        ].copy()
        
        purchase_anomalies["anomaly_type"] = np.where(
            (purchase_anomalies["Invoice Amount"] > purchase_mean + 2*purchase_std) |
            (purchase_anomalies["Invoice Amount"] < purchase_mean - 2*purchase_std),
            "Z-score",
            "IQR"
        )
    
    # Display anomalies
    anomaly_cols = st.columns(3)
    
    with anomaly_cols[0]:
        st.markdown(add_tooltip("Revenue Anomalies", "Days with unusually high or low revenue, detected using Z-score (>2 std dev) or IQR (outside 1.5x IQR range)."), unsafe_allow_html=True)
        st.markdown("##### Revenue Anomalies")
        if not filtered_data["daily_income"].empty:
            if not revenue_anomalies.empty:
                st.dataframe(revenue_anomalies[["date", "Total"]], use_container_width=True)
            else:
                st.success("No revenue anomalies detected")
        else:
            st.warning("No revenue data available")
    
    with anomaly_cols[1]:
        st.markdown(add_tooltip("Expense Anomalies", "Expenses that are unusually high or low compared to the average, detected using Z-score or IQR methods."), unsafe_allow_html=True)
        st.markdown("##### Expense Anomalies")
        if not filtered_data["expenses"].empty:
            if not expense_anomalies.empty:
                st.dataframe(expense_anomalies[["date", "Expense Amount", "Expense Type"]], use_container_width=True)
            else:
                st.success("No expense anomalies detected")
        else:
            st.warning("No expense data available")
    
    with anomaly_cols[2]:
        st.markdown(add_tooltip("Purchase Anomalies", "Inventory purchases with unusually high or low invoice amounts, detected using Z-score or IQR methods."), unsafe_allow_html=True)
        st.markdown("##### Purchase Anomalies")
        if not filtered_data["inventory"].empty:
            if not purchase_anomalies.empty:
                st.dataframe(purchase_anomalies[["date", "Invoice Amount", "Invoice Company"]], use_container_width=True)
            else:
                st.success("No purchase anomalies detected")
        else:
            st.warning("No purchase data available")
    
    # Anomaly Visualization
    st.markdown(add_tooltip("Anomaly Visualization", "Line chart showing daily revenue with markers indicating detected anomalies in revenue (red), expenses (orange), and purchases (purple)."), unsafe_allow_html=True)
    st.markdown("##### Anomaly Visualization")
    if not filtered_data["daily_income"].empty:
        fig_anomalies = go.Figure()
        
        # Add revenue trace
        fig_anomalies.add_trace(go.Scatter(
            x=filtered_data["daily_income"]["date"],
            y=filtered_data["daily_income"]["Total"],
            name="Revenue",
            mode="lines"
        ))
        
        # Add revenue anomalies
        if not revenue_anomalies.empty:
            fig_anomalies.add_trace(go.Scatter(
                x=revenue_anomalies["date"],
                y=revenue_anomalies["Total"],
                name="Revenue Anomalies",
                mode="markers",
                marker=dict(color="red", size=10)
            ))
        
        # Add expense anomalies if available
        if not filtered_data["expenses"].empty and not expense_anomalies.empty:
            fig_anomalies.add_trace(go.Scatter(
                x=expense_anomalies["date"],
                y=expense_anomalies["Expense Amount"],
                name="Expense Anomalies",
                mode="markers",
                marker=dict(color="orange", size=10)
            ))
        
        # Add purchase anomalies if available
        if not filtered_data["inventory"].empty and not purchase_anomalies.empty:
            fig_anomalies.add_trace(go.Scatter(
                x=purchase_anomalies["date"],
                y=purchase_anomalies["Invoice Amount"],
                name="Purchase Anomalies",
                mode="markers",
                marker=dict(color="purple", size=10)
            ))
        
        fig_anomalies.update_layout(
            title="Anomaly Detection",
            xaxis_title="Date",
            yaxis_title="Amount (EGP)",
            height=400
        )
        st.plotly_chart(fig_anomalies, use_container_width=True)
    
   
    
    # Inventory Turnover Analysis
    st.markdown("#### 📦 Inventory Turnover Metrics")
    
    if not filtered_data["inventory"].empty and not filtered_data["daily_income"].empty:
        # Calculate COGS (approximate as total purchases)
        cogs = filtered_data["inventory"]["Invoice Amount"].sum()
        
        # Estimate average inventory (using 30% of COGS as proxy)
        avg_inventory = cogs * 0.3
        
        # Calculate turnover metrics
        turnover_ratio = cogs / avg_inventory if avg_inventory > 0 else 0
        days_inventory = 365 / turnover_ratio if turnover_ratio > 0 else 0
        
        # Display metrics
        turnover_cols = st.columns(3)
        
        with turnover_cols[0]:
            st.markdown(add_tooltip("Inventory Turnover Ratio", "Measures how many times inventory is sold and replaced over a period (COGS / Average Inventory). Higher is generally better."), unsafe_allow_html=True)
            st.metric("", f"{turnover_ratio:.2f}x")
            st.markdown("""
                <div style="font-size: 0.9rem; color: #666; margin-top: -10px;">
                (COGS / Average Inventory)
                </div>
            """, unsafe_allow_html=True)
            
        with turnover_cols[1]:
            st.markdown(add_tooltip("Days Inventory Outstanding", "Average number of days inventory is held before being sold (365 / Turnover Ratio). Lower is generally better."), unsafe_allow_html=True)
            st.metric("", f"{days_inventory:.1f} days")
            st.markdown("""
                <div style="font-size: 0.9rem; color: #666; margin-top: -10px;">
                (365 / Turnover Ratio)
                </div>
            """, unsafe_allow_html=True)
            
        with turnover_cols[2]:
            st.markdown(add_tooltip("COGS (Estimated)", "Cost of Goods Sold, estimated as the total amount spent on inventory purchases."), unsafe_allow_html=True)
            st.metric("", f"EGP {cogs:,.2f}")
            st.markdown("""
                <div style="font-size: 0.9rem; color: #666; margin-top: -10px;">
                (Total Purchases)
                </div>
            """, unsafe_allow_html=True)


        # Turnover insights
        st.markdown("##### 🎯 Turnover Insights")
        
        if turnover_ratio > 6:
            st.success("""
                **High Turnover**: Inventory is selling quickly, indicating strong demand and efficient inventory management. 
                Consider increasing stock levels of fast-moving items.
            """)
        elif turnover_ratio > 3:
            st.info("""
                **Healthy Turnover**: Inventory is moving at a good pace. Maintain current ordering patterns.
            """)
        else:
            st.warning("""
                **Low Turnover**: Inventory is moving slowly, which may indicate overstocking or weak demand. 
                Consider promotions or reducing order quantities.
            """)        
        # Monthly turnover trends
        st.markdown(add_tooltip("Monthly Turnover Trends", "Line chart showing the calculated inventory turnover ratio aggregated by month."), unsafe_allow_html=True)
        st.markdown("##### 📈 Monthly Turnover Trends")
        
        # Calculate monthly COGS and estimate monthly inventory
        monthly_cogs = filtered_data["inventory"].groupby(
            filtered_data["inventory"]["date"].dt.strftime('%Y-%m')
        )["Invoice Amount"].sum()
        
        monthly_avg_inventory = monthly_cogs * 0.3
        monthly_turnover = monthly_cogs / monthly_avg_inventory
        
        # Create visualization
        fig_turnover = go.Figure()
        fig_turnover.add_trace(go.Scatter(
            x=monthly_turnover.index,
            y=monthly_turnover.values,
            name="Turnover Ratio",
            line=dict(color=COLOR_PALETTE['primary'])
        ))
        fig_turnover.update_layout(
            title="", # Title provided by markdown tooltip
            xaxis_title="Month",
            yaxis_title="Turnover Ratio",
            height=300
        )
        st.plotly_chart(fig_turnover, use_container_width=True)
        
 # Time Series Analysis
    st.markdown("#### 📈 Time Series Analysis")
    ts_tabs = st.tabs(["Decomposition", "Statistical Tests", "Autocorrelation", "ARIMA Modeling"])
    
    with ts_tabs[0]:
        st.markdown(add_tooltip("Time Series Decomposition", "Breaks down the selected metric's time series into observed, trend, seasonal, and residual components."), unsafe_allow_html=True)
        # Select metric for decomposition
        decomp_metric = st.selectbox(
            "Select Metric for Decomposition",
            ["Daily Revenue", "Daily Expenses", "Daily Purchases"],
            key="decomp_metric"
        )
        
        # Prepare time series data
        if decomp_metric == "Daily Revenue":
            ts_data = filtered_data["daily_income"].set_index("date")["Total"]
        elif decomp_metric == "Daily Expenses":
            ts_data = filtered_data["expenses"].groupby("date")["Expense Amount"].sum()
        else:
            ts_data = filtered_data["inventory"].groupby("date")["Invoice Amount"].sum()

        # Ensure continuous daily data and sufficient observations
        date_range = pd.date_range(start=ts_data.index.min(), end=ts_data.index.max(), freq='D')
        ts_data = ts_data.reindex(date_range).fillna(method='ffill').fillna(method='bfill')
        
        
        if len(ts_data) >= 14:  # Minimum required for 2 complete cycles with period=7
            try:
                # Perform decomposition
                decomposition = seasonal_decompose(ts_data, period=7, extrapolate_trend='freq')
                # Display decomposition statistics
                st.markdown("#### Decomposition Statistics")
                stats_cols = st.columns(4)
                
                with stats_cols[0]:
                    st.markdown(add_tooltip("Trend Strength", "Measures how much of the data's variance is explained by the trend component."), unsafe_allow_html=True)
                    st.metric("", 
                             f"{(1 - np.var(decomposition.resid)/np.var(decomposition.trend + decomposition.resid))*100:.1f}%")
                
                with stats_cols[1]:
                    st.markdown(add_tooltip("Seasonal Strength", "Measures how much of the data's variance is explained by the seasonal component."), unsafe_allow_html=True)
                    st.metric("", 
                             f"{(1 - np.var(decomposition.resid)/np.var(decomposition.seasonal + decomposition.resid))*100:.1f}%")
                
                with stats_cols[2]:
                    st.markdown(add_tooltip("Residual Volatility", "Standard deviation of the residuals as a percentage of the mean observed value. Indicates unexplained variability."), unsafe_allow_html=True)
                    st.metric("",
                             f"{np.std(decomposition.resid)/np.mean(decomposition.observed)*100:.1f}%")
                
                with stats_cols[3]:
                    st.markdown(add_tooltip("Data Points", "Number of data points used in the decomposition analysis."), unsafe_allow_html=True)
                    st.metric("",
                             f"{len(ts_data)}")
                # Plot decomposition components
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
                
                decomposition.observed.plot(ax=ax1)
                ax1.set_title('Observed')
                decomposition.trend.plot(ax=ax2)
                ax2.set_title('Trend')
                decomposition.seasonal.plot(ax=ax3)
                ax3.set_title('Seasonal')
                decomposition.resid.plot(ax=ax4)
                ax4.set_title('Residual')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
               
            except Exception as e:
                st.error(f"Error in decomposition: {str(e)}")
                st.write("Try adjusting the date range to include more data points.")
        else:
            st.warning(f"Insufficient data for seasonal decomposition. Need at least 14 days, but got {len(ts_data)} days.")
            
            # Display basic time series plot instead
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts_data.index,
                y=ts_data.values,
                mode='lines+markers',
                name='Observed Values'
            ))
            fig.update_layout(
                title=f"{decomp_metric} Time Series",
                xaxis_title="Date",
                yaxis_title="Value",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
    with ts_tabs[1]:
        # Statistical Tests
        st.markdown(add_tooltip("Stationarity Tests", "Performs Augmented Dickey-Fuller (ADF) and KPSS tests to check if the time series is stationary (mean and variance are constant over time). Stationarity is often required for time series modeling."), unsafe_allow_html=True)
        st.markdown("##### Stationarity Tests")
        test_metric = st.selectbox(
            "Select Metric for Testing",
            ["Daily Revenue", "Daily Expenses", "Daily Purchases"],
            key="test_metric"
        )
        
        # Prepare data for testing
        if test_metric == "Daily Revenue":
            test_data = filtered_data["daily_income"].set_index("date")["Total"]
        elif test_metric == "Daily Expenses":
            test_data = filtered_data["expenses"].groupby("date")["Expense Amount"].sum()
        else:
            test_data = filtered_data["inventory"].groupby("date")["Invoice Amount"].sum()
        
        # Augmented Dickey-Fuller test
        adf_result = adfuller(test_data.dropna())
        st.write("Augmented Dickey-Fuller Test:")
        st.write(f"ADF Statistic: {adf_result[0]:.4f}")
        st.write(f"p-value: {adf_result[1]:.4f}")
        st.write("Critical Values:")
        for key, value in adf_result[4].items():
            st.write(f"\t{key}: {value:.4f}")
            
        # KPSS test
        kpss_result = kpss(test_data.dropna())
        st.write("\nKPSS Test:")
        st.write(f"KPSS Statistic: {kpss_result[0]:.4f}")
        st.write(f"p-value: {kpss_result[1]:.4f}")
        st.write("Critical Values:")
        for key, value in kpss_result[3].items():
            st.write(f"\t{key}: {value:.4f}")
            
    with ts_tabs[2]:
        # Autocorrelation Analysis
        st.markdown(add_tooltip("Autocorrelation Analysis", "Plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) to identify patterns and dependencies at different time lags. Useful for determining ARIMA model orders."), unsafe_allow_html=True)
        st.markdown("##### Autocorrelation Analysis")
        acf_metric = st.selectbox(
            "Select Metric for Autocorrelation",
            ["Daily Revenue", "Daily Expenses", "Daily Purchases"],
            key="acf_metric"
        )
        
        max_lags = st.slider("Maximum Lags", 5, 50, 20)
        
        # Prepare data
        if acf_metric == "Daily Revenue":
            acf_data = filtered_data["daily_income"].set_index("date")["Total"]
        elif acf_metric == "Daily Expenses":
            acf_data = filtered_data["expenses"].groupby("date")["Expense Amount"].sum()
        else:
            acf_data = filtered_data["inventory"].groupby("date")["Invoice Amount"].sum()
        
        # Ensure data is continuous and properly sorted
        date_range = pd.date_range(start=acf_data.index.min(), end=acf_data.index.max(), freq='D')
        acf_data = acf_data.reindex(date_range).fillna(method='ffill').fillna(method='bfill')
        
        try:
            # Create figure with more space between subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            plt.subplots_adjust(hspace=0.3)
            
            # Calculate and plot ACF
            acf_values = sm.tsa.stattools.acf(acf_data, nlags=max_lags, fft=True)
            lags = np.arange(len(acf_values))
            ax1.bar(lags, acf_values, width=0.3)
            ax1.axhline(y=0, linestyle='-', color='black', alpha=0.5)
            ax1.axhline(y=1.96/np.sqrt(len(acf_data)), linestyle='--', color='gray', alpha=0.5)
            ax1.axhline(y=-1.96/np.sqrt(len(acf_data)), linestyle='--', color='gray', alpha=0.5)
            ax1.set_title('Autocorrelation Function')
            ax1.set_xlabel('Lag')
            ax1.set_ylabel('Correlation')
            
            # Calculate and plot PACF
            pacf_values = sm.tsa.stattools.pacf(acf_data, nlags=max_lags, method='ols')
            lags = np.arange(len(pacf_values))
            ax2.bar(lags, pacf_values, width=0.3)
            ax2.axhline(y=0, linestyle='-', color='black', alpha=0.5)
            ax2.axhline(y=1.96/np.sqrt(len(acf_data)), linestyle='--', color='gray', alpha=0.5)
            ax2.axhline(y=-1.96/np.sqrt(len(acf_data)), linestyle='--', color='gray', alpha=0.5)
            ax2.set_title('Partial Autocorrelation Function')
            ax2.set_xlabel('Lag')
            ax2.set_ylabel('Correlation')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Add correlation interpretation
            st.markdown("##### Correlation Interpretation")
            
            # ACF Analysis
            significant_acf = np.where(np.abs(acf_values) > 1.96/np.sqrt(len(acf_data)))[0]
            if len(significant_acf) > 0:
                st.write(f"Significant autocorrelations found at lags: {', '.join(map(str, significant_acf))}")
                if 1 in significant_acf:
                    st.write("Strong day-to-day correlation detected")
                if 7 in significant_acf:
                    st.write("Weekly seasonal pattern detected")
            else:
                st.write("No significant autocorrelations found")
            
            # PACF Analysis
            significant_pacf = np.where(np.abs(pacf_values) > 1.96/np.sqrt(len(acf_data)))[0]
            if len(significant_pacf) > 0:
                st.write(f"Significant partial autocorrelations found at lags: {', '.join(map(str, significant_pacf))}")
                suggested_ar = max(significant_pacf)
                st.write(f"Suggested AR order for ARIMA model: {suggested_ar}")
            else:
                st.write("No significant partial autocorrelations found")
                
        except Exception as e:
            st.error(f"Error calculating correlations: {str(e)}")
            st.write("Try adjusting the date range or selecting a different metric.")

    with ts_tabs[3]:
        # ARIMA Modeling
        st.markdown(add_tooltip("ARIMA Modeling", "Fits an Autoregressive Integrated Moving Average (ARIMA) model to the selected time series data based on the specified (p, d, q) orders. Shows model summary and fitted vs actual values."), unsafe_allow_html=True)
        st.markdown("##### ARIMA Model")
        arima_metric = st.selectbox(
            "Select Metric for ARIMA",
            ["Daily Revenue", "Daily Expenses", "Daily Purchases"],
            key="arima_metric"
        )
        
        cols = st.columns(3)
        p = cols[0].number_input("AR(p)", 0, 5, 1)
        d = cols[1].number_input("I(d)", 0, 2, 1)
        q = cols[2].number_input("MA(q)", 0, 5, 1)
        
        # Prepare data
        if arima_metric == "Daily Revenue":
            arima_data = filtered_data["daily_income"].set_index("date")["Total"]
        elif arima_metric == "Daily Expenses":
            arima_data = filtered_data["expenses"].groupby("date")["Expense Amount"].sum()
        else:
            arima_data = filtered_data["inventory"].groupby("date")["Invoice Amount"].sum()
            
        # Fit ARIMA model
        model = ARIMA(arima_data, order=(p, d, q))
        results = model.fit()
        
        # Display results
        st.write("Model Summary:")
        st.text(results.summary().tables[1].as_text())
        
        # Plot results
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=arima_data.index,
            y=arima_data.values,
            name="Actual",
            line=dict(color=COLOR_PALETTE["primary"])
        ))
        fig.add_trace(go.Scatter(
            x=arima_data.index,
            y=results.fittedvalues,
            name="Fitted",
            line=dict(color=COLOR_PALETTE["secondary"])
        ))
        fig.update_layout(
            title="ARIMA Model Fit",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Data Tables
    st.markdown("#### 📋 Detailed Data Tables")
    tab1, tab2, tab3 = st.tabs(["Daily Income", "Inventory", "Expenses"])
    
    with tab1:
        st.dataframe(
            filtered_data["daily_income"].sort_values("date", ascending=False),
            use_container_width=True
        )
    
    with tab2:
        st.dataframe(
            filtered_data["inventory"].sort_values("date", ascending=False),
            use_container_width=True
        )
    
    with tab3:
        st.dataframe(
            filtered_data["expenses"].sort_values("date", ascending=False),
            use_container_width=True
        )

# ML & Predictions Tab
with tab_ml:
    st.markdown(add_tooltip("🤖 Advanced Analytics & Predictions", 
                            "Utilizes machine learning models (Prophet) to forecast key financial metrics and analyze time series patterns."), 
                unsafe_allow_html=True)
    
    # Setup prediction data
    pred_cols = st.columns([2, 1])
    with pred_cols[1]:
        prediction_days = st.slider("Prediction Days", 7, 90, 30, 
                                    help="Number of future days to forecast.")
        confidence_interval = st.slider("Confidence Interval", 0.8, 0.99, 0.95, 
                                        help="The probability range for the forecast values (e.g., 0.95 means 95% confidence).")
        
    # Prepare integrated dataset for predictions
    # ... (data preparation code remains the same) ...
    daily_metrics = pd.DataFrame()
    daily_metrics['date'] = filtered_data["daily_income"]["date"]
    daily_metrics['revenue'] = filtered_data["daily_income"]["Total"]
    
    # Ensure expenses and purchases are aligned with revenue dates
    expenses_agg = filtered_data["expenses"].groupby("date")["Expense Amount"].sum().reindex(daily_metrics['date']).fillna(0)
    purchases_agg = filtered_data["inventory"].groupby("date")["Invoice Amount"].sum().reindex(daily_metrics['date']).fillna(0)
    
    daily_metrics['expenses'] = expenses_agg.values
    daily_metrics['purchases'] = purchases_agg.values
    daily_metrics['deficit'] = filtered_data["daily_income"]["deficit"]
    daily_metrics['cash'] = filtered_data["daily_income"]["cash"]
    daily_metrics['visa'] = filtered_data["daily_income"]["visa"]
    daily_metrics['due_amount'] = filtered_data["daily_income"]["due amount"]
    daily_metrics = daily_metrics.fillna(0)
    
    # Calculate derived metrics
    daily_metrics['net_profit'] = daily_metrics['revenue'] - daily_metrics['expenses'] - daily_metrics['purchases']
    daily_metrics['profit_margin'] = (daily_metrics['net_profit'] / daily_metrics['revenue']).fillna(0)
    daily_metrics['expense_ratio'] = (daily_metrics['expenses'] / daily_metrics['revenue']).fillna(0)

    # Multi-metric Prophet Models
    # ... (model training loop remains the same) ...
    metrics_to_predict = {
        'Revenue': daily_metrics['revenue'],
        'Expenses': daily_metrics['expenses'],
        'Purchases': daily_metrics['purchases'],
        'Net Profit': daily_metrics['net_profit']
    }
    
    forecast_results = {}
    model_metrics = {}
    
    with st.spinner("Training multiple prediction models..."):
        for metric_name, metric_data in metrics_to_predict.items():
            # Prepare data
            df_prophet = pd.DataFrame({
                'ds': daily_metrics['date'],
                'y': metric_data
            })
            
            # Configure model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                changepoint_prior_scale=0.05,
                interval_width=confidence_interval
            )
            
            # Add custom seasonality
            model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=5
            )
            
            # Fit model
            model.fit(df_prophet)
            
            # Make future predictions
            future = model.make_future_dataframe(periods=prediction_days)
            forecast = model.predict(future)
            
            forecast_results[metric_name] = {
                'model': model,
                'forecast': forecast,
                'actual': df_prophet
            }
            
            # Calculate model metrics
            train_rmse = np.sqrt(mean_squared_error(
                df_prophet['y'],
                forecast['yhat'][:len(df_prophet)]
            ))
            model_metrics[metric_name] = {
                'rmse': train_rmse,
                'accuracy': 1 - (train_rmse / df_prophet['y'].mean()) if df_prophet['y'].mean() != 0 else np.nan
            }

    # Display Model Performance Metrics
    st.markdown(add_tooltip("📊 Model Performance", 
                            "Accuracy and Root Mean Squared Error (RMSE) for each prediction model. Accuracy is 1 - (RMSE / Mean Actual Value). Higher accuracy and lower RMSE are better."), 
                unsafe_allow_html=True)
    metric_cols = st.columns(len(model_metrics))
    for idx, (metric_name, metrics) in enumerate(model_metrics.items()):
        with metric_cols[idx]:
            st.metric(
                f"{metric_name} Model Accuracy",
                f"{metrics['accuracy']*100:.1f}%" if not np.isnan(metrics['accuracy']) else "N/A",
                f"RMSE: {metrics['rmse']:,.2f}",
                help=f"Accuracy: 1 - (RMSE / Mean {metric_name}). RMSE: Root Mean Squared Error, lower is better."
            )
    
    # Integrated Forecast Visualization
    st.markdown(add_tooltip("📈 Integrated Forecasts", 
                            "Visualizations of actual vs. forecasted values, seasonality patterns, and correlations between metrics."), 
                unsafe_allow_html=True)
    
    forecast_tabs = st.tabs([
        "Revenue & Profit",
        "Expenses & Purchases",
        "Seasonality Analysis",
        "Correlation Analysis"
    ])
    
    with forecast_tabs[0]:
        st.markdown(add_tooltip("Revenue and Profit Forecasts", 
                                "Shows actual vs. forecasted Revenue and Net Profit, including confidence intervals."), 
                    unsafe_allow_html=True)
        fig = go.Figure()
        
        for metric in ['Revenue', 'Net Profit']:
            forecast = forecast_results[metric]['forecast']
            actual = forecast_results[metric]['actual']
            
            # Actual values
            fig.add_trace(go.Scatter(
                x=actual['ds'],
                y=actual['y'],
                name=f'Actual {metric}',
                line=dict(color=COLOR_PALETTE['primary'] if metric == 'Revenue' else COLOR_PALETTE['accent']),
                hovertemplate=f'<b>{metric} (Actual)</b><br>Date: %{{x|%Y-%m-%d}}<br>Amount: %{{y:,.2f}} EGP<extra></extra>'
            ))
            
            # Forecast values
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                name=f'Forecast {metric}',
                line=dict(dash='dash', color=COLOR_PALETTE['primary'] if metric == 'Revenue' else COLOR_PALETTE['accent']),
                hovertemplate=f'<b>{metric} (Forecast)</b><br>Date: %{{x|%Y-%m-%d}}<br>Amount: %{{y:,.2f}} EGP<extra></extra>'
            ))
            
            # Confidence intervals
            fig.add_trace(go.Scatter(
                x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                fill='toself',
                fillcolor=f'rgba{tuple(list(int(COLOR_PALETTE["primary"].lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}' if metric == 'Revenue' else f'rgba{tuple(list(int(COLOR_PALETTE["accent"].lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{metric} Confidence Interval',
                hoverinfo='skip' # Hide hover for the fill area itself
            ))
        
        fig.update_layout(
            title="Revenue and Profit Forecast",
            xaxis_title="Date",
            yaxis_title="Amount (EGP)",
            height=500,
            hovermode='x unified' # Show unified hover info
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Key Metrics
        st.markdown(add_tooltip("Key Forecast Metrics", 
                                "Compares the last actual value to the last forecasted value (Growth) and the average forecast trend over the prediction period."), 
                    unsafe_allow_html=True)
        metric_cols = st.columns(4)
        for idx, metric in enumerate(['Revenue', 'Net Profit']):
            forecast = forecast_results[metric]['forecast']
            last_actual = forecast_results[metric]['actual']['y'].iloc[-1]
            last_forecast = forecast['yhat'].iloc[-1]
            
            with metric_cols[idx*2]:
                growth_rate = ((last_forecast - last_actual) / last_actual) * 100 if last_actual != 0 else 0
                st.metric(
                    f"{metric} Growth",
                    f"{growth_rate:.1f}%",
                    f"EGP {last_forecast - last_actual:,.2f}",
                    help=f"Percentage change from the last actual {metric} to the last forecasted {metric}."
                )
            
            with metric_cols[idx*2 + 1]:
                forecast_avg = forecast['yhat'].tail(prediction_days).mean()
                current_avg = forecast_results[metric]['actual']['y'].tail(prediction_days).mean()
                trend_change_pct = ((forecast_avg - current_avg) / current_avg * 100) if current_avg != 0 else 0
                st.metric(
                    f"{metric} Trend",
                    f"EGP {forecast_avg:,.2f}",
                    f"{trend_change_pct:.1f}%",
                    help=f"Average forecasted {metric} over the next {prediction_days} days compared to the average of the last {prediction_days} actual days."
                )
    
    with forecast_tabs[1]:
        st.markdown(add_tooltip("Expenses and Purchases Forecasts", 
                                "Shows actual vs. forecasted Expenses and Purchases."), 
                    unsafe_allow_html=True)
        fig = go.Figure()
        
        for metric in ['Expenses', 'Purchases']:
            forecast = forecast_results[metric]['forecast']
            actual = forecast_results[metric]['actual']
            
            fig.add_trace(go.Scatter(
                x=actual['ds'],
                y=actual['y'],
                name=f'Actual {metric}',
                line=dict(color=COLOR_PALETTE['secondary'] if metric == 'Expenses' else COLOR_PALETTE['neutral']),
                hovertemplate=f'<b>{metric} (Actual)</b><br>Date: %{{x|%Y-%m-%d}}<br>Amount: %{{y:,.2f}} EGP<extra></extra>'
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                name=f'Forecast {metric}',
                line=dict(dash='dash', color=COLOR_PALETTE['secondary'] if metric == 'Expenses' else COLOR_PALETTE['neutral']),
                hovertemplate=f'<b>{metric} (Forecast)</b><br>Date: %{{x|%Y-%m-%d}}<br>Amount: %{{y:,.2f}} EGP<extra></extra>'
            ))
            
            # Confidence intervals (Optional, can add if needed like in Tab 0)
            fig.add_trace(go.Scatter(
                x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                fill='toself',
                fillcolor=f'rgba{tuple(list(int(COLOR_PALETTE["secondary"].lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}' if metric == 'Expenses' else f'rgba{tuple(list(int(COLOR_PALETTE["neutral"].lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{metric} Confidence Interval',
                hoverinfo='skip'
            ))

        fig.update_layout(
            title="Expenses and Purchases Forecast",
            xaxis_title="Date",
            yaxis_title="Amount (EGP)",
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast Impact Analysis
        st.markdown(add_tooltip("Forecast Impact Analysis", 
                                "Estimates the total cost impact and cost ratio based on the forecasts."), 
                    unsafe_allow_html=True)
        impact_cols = st.columns(2)
        
        with impact_cols[0]:
            expenses_forecast = forecast_results['Expenses']['forecast']
            purchases_forecast = forecast_results['Purchases']['forecast']
            
            total_cost_forecast = expenses_forecast['yhat'].tail(prediction_days).sum() + \
                                purchases_forecast['yhat'].tail(prediction_days).sum()
            
            current_total_cost = daily_metrics['expenses'].tail(prediction_days).sum() + \
                               daily_metrics['purchases'].tail(prediction_days).sum()
            
            cost_impact_pct = ((total_cost_forecast - current_total_cost) / current_total_cost * 100) if current_total_cost != 0 else 0
            st.metric(
                "Projected Cost Impact",
                f"EGP {total_cost_forecast:,.2f}",
                f"{cost_impact_pct:.1f}%",
                help=f"Total projected Expenses + Purchases over the next {prediction_days} days compared to the last {prediction_days} days."
            )
        
        with impact_cols[1]:
            revenue_forecast_sum = forecast_results['Revenue']['forecast']['yhat'].tail(prediction_days).sum()
            cost_ratio_forecast = total_cost_forecast / revenue_forecast_sum if revenue_forecast_sum != 0 else 0
            
            current_revenue_sum = daily_metrics['revenue'].tail(prediction_days).sum()
            current_cost_ratio = current_total_cost / current_revenue_sum if current_revenue_sum != 0 else 0
            
            cost_ratio_change = (cost_ratio_forecast - current_cost_ratio) * 100
            st.metric(
                "Projected Cost Ratio",
                f"{cost_ratio_forecast:.1%}",
                f"{cost_ratio_change:.1f} % points", # Changed delta format for clarity
                help=f"Projected (Expenses + Purchases) / Revenue ratio over the next {prediction_days} days, compared to the current ratio."
            )
    
    with forecast_tabs[2]:
        st.markdown(add_tooltip("Seasonality Analysis", 
                                "Analyzes recurring patterns (yearly, weekly, daily, monthly) identified by the Prophet model for each metric."), 
                    unsafe_allow_html=True)
        for metric_name, metric_results in forecast_results.items():
            model = metric_results['model']
            
            st.markdown(add_tooltip(f"{metric_name} Seasonality Components", 
                                    f"Shows the estimated impact of different time-based patterns (trend, weekly, yearly, monthly) on {metric_name}."), 
                        unsafe_allow_html=True)
            
            # Plot seasonality components
            try:
                fig_components = model.plot_components(metric_results['forecast'])
                # Add tooltips/titles to subplots if possible (Matplotlib makes this harder than Plotly)
                axes = fig_components.get_axes()
                if len(axes) > 0: axes[0].set_title(f"{metric_name} - Trend Component")
                if len(axes) > 1: axes[1].set_title(f"{metric_name} - Weekly Seasonality")
                if len(axes) > 2: axes[2].set_title(f"{metric_name} - Yearly Seasonality")
                if len(axes) > 3: axes[3].set_title(f"{metric_name} - Monthly Seasonality")
                fig_components.tight_layout()
                st.pyplot(fig_components)
                plt.close(fig_components) # Close the figure to free memory
            except Exception as e:
                st.warning(f"Could not plot components for {metric_name}: {e}")

            # Create mapping for metric names to column names
            metric_mapping = {
                'Revenue': 'revenue',
                'Expenses': 'expenses',
                'Purchases': 'purchases',
                'Net Profit': 'net_profit'
            }
            
            # Weekly patterns using correct column names
            metric_col = metric_mapping.get(metric_name)
            if metric_col and metric_col in daily_metrics.columns:
                st.markdown(add_tooltip(f"{metric_name} - Average by Day of Week", 
                                        f"Shows the average {metric_name} for each day of the week based on historical data."), 
                            unsafe_allow_html=True)
                weekly_pattern = daily_metrics.groupby(daily_metrics['date'].dt.day_name())[metric_col].mean()
                # Ensure correct order of days
                ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                weekly_pattern = weekly_pattern.reindex(ordered_days)

                fig_weekly = go.Figure(data=[go.Bar(
                    x=weekly_pattern.index,
                    y=weekly_pattern.values,
                    marker_color=COLOR_PALETTE['primary'],
                    hovertemplate='<b>Day:</b> %{x}<br><b>Average Amount:</b> %{y:,.2f} EGP<extra></extra>'
                )])
                
                fig_weekly.update_layout(
                    title=f"{metric_name} - Average by Day of Week",
                    xaxis_title="Day of Week",
                    yaxis_title="Amount (EGP)",
                    height=300
                )
                st.plotly_chart(fig_weekly, use_container_width=True)
            else:
                st.warning(f"Could not find column '{metric_col}' for weekly pattern analysis of {metric_name}.")

    with forecast_tabs[3]:
        st.markdown(add_tooltip("🔄 Metric Correlations", 
                                "Analyzes the relationships between different financial metrics."), 
                    unsafe_allow_html=True)
        
        # Calculate correlations
        metrics_for_corr = ['revenue', 'expenses', 'purchases', 'net_profit']
        # Ensure columns exist before calculating correlation
        valid_metrics_for_corr = [m for m in metrics_for_corr if m in daily_metrics.columns]
        if len(valid_metrics_for_corr) > 1:
            correlation_matrix = daily_metrics[valid_metrics_for_corr].corr()
            
            st.markdown(add_tooltip("Correlation Matrix", 
                                    "Heatmap showing the Pearson correlation coefficient between pairs of metrics. Values range from -1 (strong negative correlation) to +1 (strong positive correlation). 0 indicates no linear correlation."), 
                        unsafe_allow_html=True)
            fig_corr = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                hovertemplate='<b>X:</b> %{x}<br><b>Y:</b> %{y}<br><b>Correlation:</b> %{z:.2f}<extra></extra>',
                colorbar=dict(title='Correlation')
            ))
            
            fig_corr.update_layout(
                title="Correlation Matrix",
                height=400
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Cross-correlation analysis
            st.markdown(add_tooltip("Cross-Correlation Analysis", 
                                    "Examines the correlation between two metrics at different time lags. Helps identify leading or lagging relationships (e.g., does an increase in purchases lead to an increase in revenue a few days later?)."), 
                        unsafe_allow_html=True)
            lag_range = st.slider("Lag Range (Days)", -30, 30, (-7, 7), 
                                  help="The range of time lags (in days) to check for correlations. Negative lag means the second metric leads the reference metric.")
            
            reference_metric = st.selectbox(
                "Reference Metric",
                options=valid_metrics_for_corr,
                help="The primary metric to compare others against."
            )
            
            for metric in valid_metrics_for_corr:
                if metric != reference_metric:
                    ccf = pd.DataFrame(
                        index=range(lag_range[0], lag_range[1] + 1),
                        columns=['correlation']
                    )
                    
                    for lag in range(lag_range[0], lag_range[1] + 1):
                        try:
                            if lag < 0:
                                # Reference metric lagged behind 'metric'
                                ccf.loc[lag, 'correlation'] = daily_metrics[reference_metric].corr(
                                    daily_metrics[metric].shift(-lag) # Shift 'metric' forward
                                )
                            else:
                                # Reference metric leading 'metric'
                                ccf.loc[lag, 'correlation'] = daily_metrics[reference_metric].corr(
                                    daily_metrics[metric].shift(lag) # Shift 'metric' backward
                                )
                        except Exception as e:
                            st.warning(f"Could not calculate cross-correlation for lag {lag} between {reference_metric} and {metric}: {e}")
                            ccf.loc[lag, 'correlation'] = np.nan

                    ccf = ccf.dropna() # Drop rows where correlation couldn't be calculated

                    if not ccf.empty:
                        fig_ccf = go.Figure(data=go.Bar(
                            x=ccf.index,
                            y=ccf['correlation'],
                            marker_color=COLOR_PALETTE['primary'],
                            hovertemplate='<b>Lag:</b> %{x} days<br><b>Correlation:</b> %{y:.2f}<extra></extra>'
                        ))
                        
                        fig_ccf.update_layout(
                            title=f"Cross-correlation: {reference_metric.title()} vs {metric.title()}",
                            xaxis_title="Lag (Days)",
                            yaxis_title="Correlation",
                            height=300
                        )
                        st.plotly_chart(fig_ccf, use_container_width=True)
                    else:
                         st.warning(f"Not enough overlapping data to calculate cross-correlation between {reference_metric} and {metric} for the selected lag range.")

        else:
            st.warning("Insufficient metrics available in the dataset for correlation analysis.")

# Search & Reports Tab
with tab_search:
    st.markdown(add_tooltip("🔍 Inventory Purchase Search & Reports", 
                            "Search through inventory purchase records using various filters and generate reports."), 
                unsafe_allow_html=True)
    
    # Search KPIs
    st.markdown(add_tooltip("📊 Search Results Summary (Initial)", 
                            "Summary metrics for the currently selected date range before applying search filters."), 
                unsafe_allow_html=True)
    search_kpi_cols = st.columns(4)
    
    # Calculate initial KPIs based on filtered_data["inventory"]
    initial_total_purchases = len(filtered_data["inventory"])
    initial_total_amount = filtered_data["inventory"]["Invoice Amount"].sum()
    initial_avg_purchase = filtered_data["inventory"]["Invoice Amount"].mean() if initial_total_purchases > 0 else 0
    initial_unique_companies = filtered_data["inventory"]["Invoice Company"].nunique()

    with search_kpi_cols[0]:
        st.metric("Total Purchases", f"{initial_total_purchases:,}", 
                  help="Total number of purchase invoices in the selected date range.")
    
    with search_kpi_cols[1]:
        st.metric("Total Amount", f"EGP {initial_total_amount:,.2f}", 
                  help="Total value of all purchase invoices in the selected date range.")
    
    with search_kpi_cols[2]:
        st.metric("Average Purchase", f"EGP {initial_avg_purchase:,.2f}", 
                  help="Average value per purchase invoice in the selected date range.")
    
    with search_kpi_cols[3]:
        st.metric("Unique Companies", f"{initial_unique_companies}", 
                  help="Number of unique companies from which purchases were made in the selected date range.")
    
    # Search Filters
    st.markdown(add_tooltip("🔎 Search Filters", 
                            "Apply filters to narrow down the inventory purchase records."), 
                unsafe_allow_html=True)
    search_cols = st.columns(5)
    
    with search_cols[0]:
        invoice_id = st.text_input(
            "Invoice ID",
            placeholder="Enter Invoice ID",
            help="Search for a specific invoice by its ID (partial matches allowed)."
        )
    
    with search_cols[1]:
        search_company = st.selectbox(
            "Company",
            ["All"] + sorted(filtered_data["inventory"]["Invoice Company"].unique().tolist()),
            help="Filter purchases by the supplier company."
        )
    
    with search_cols[2]:
        search_type = st.selectbox(
            "Inventory Type",
            ["All"] + sorted(filtered_data["inventory"]["Inventory Type"].unique().tolist()),
            help="Filter purchases by the type of inventory."
        )
    
    with search_cols[3]:
        # Ensure max_value is valid even if data is empty
        max_inv_amount = float(filtered_data["inventory"]["Invoice Amount"].max()) if not filtered_data["inventory"].empty else 100000.0
        min_amount = st.number_input(
            "Min Amount (EGP)",
            min_value=0.0,
            max_value=max_inv_amount,
            value=0.0,
            help="Filter purchases with an invoice amount greater than or equal to this value."
        )
    
    with search_cols[4]:
        max_amount = st.number_input(
            "Max Amount (EGP)",
            min_value=0.0,
            max_value=max_inv_amount,
            value=max_inv_amount,
            help="Filter purchases with an invoice amount less than or equal to this value."
        )
    
    # Apply filters
    search_results = filtered_data["inventory"].copy()
    
    # Apply Invoice ID filter if provided
    if invoice_id:
        search_results = search_results[
            search_results["Invoice ID"].astype(str).str.contains(invoice_id, case=False, na=False)
        ]
    
    if search_company != "All":
        search_results = search_results[search_results["Invoice Company"] == search_company]
    
    if search_type != "All":
        search_results = search_results[search_results["Inventory Type"] == search_type]
    
    # Ensure min_amount <= max_amount before filtering
    if min_amount <= max_amount:
        search_results = search_results[
            (search_results["Invoice Amount"] >= min_amount) &
            (search_results["Invoice Amount"] <= max_amount)
        ]
    else:
        st.warning("Min Amount cannot be greater than Max Amount. Please adjust the amount filters.")
        # Optionally, reset search_results or handle as appropriate
        # search_results = pd.DataFrame(columns=filtered_data["inventory"].columns) # Example: show empty results

    # Display search results
    st.markdown(add_tooltip("📋 Search Results", 
                            "Table displaying the inventory purchase records matching the applied filters."), 
                unsafe_allow_html=True)
    
    # Results summary based on search_results
    st.markdown(add_tooltip("📊 Filtered Results Summary", 
                            "Summary metrics for the purchase records matching the current search filters."), 
                unsafe_allow_html=True)
    results_cols = st.columns(3)
    
    filtered_total_purchases = len(search_results)
    filtered_total_amount = search_results["Invoice Amount"].sum()
    filtered_avg_purchase = search_results["Invoice Amount"].mean() if filtered_total_purchases > 0 else 0

    with results_cols[0]:
        st.metric(
            "Filtered Purchases",
            f"{filtered_total_purchases:,}",
            delta=f"{filtered_total_purchases - initial_total_purchases:,}",
            help="Total number of purchase invoices matching the filters. Delta shows the change from the initial total."
        )
    
    with results_cols[1]:
        st.metric(
            "Filtered Amount",
            f"EGP {filtered_total_amount:,.2f}",
            delta=f"EGP {filtered_total_amount - initial_total_amount:,.2f}",
            help="Total value of purchase invoices matching the filters. Delta shows the change from the initial total amount."
        )
    
    with results_cols[2]:
        st.metric(
            "Filtered Average",
            f"EGP {filtered_avg_purchase:,.2f}",
            delta=f"EGP {filtered_avg_purchase - initial_avg_purchase:,.2f}",
            help="Average value per purchase invoice matching the filters. Delta shows the change from the initial average."
        )
    
    # Detailed results table
    st.dataframe(
        search_results.sort_values("date", ascending=False),
        use_container_width=True
    )
    
    # Purchase Analysis based on search_results
    st.markdown(add_tooltip("📊 Purchase Analysis (Filtered)", 
                            "Visual analysis of the filtered purchase records."), 
                unsafe_allow_html=True)
    analysis_cols = st.columns(2)
    
    with analysis_cols[0]:
        st.markdown(add_tooltip("Purchases by Company (Filtered)", 
                                "Distribution of total purchase amounts by company for the filtered results."), 
                    unsafe_allow_html=True)
        if not search_results.empty:
            company_dist = search_results.groupby("Invoice Company")["Invoice Amount"].sum().sort_values(ascending=True)
            fig_company = go.Figure(data=[go.Bar(
                x=company_dist.values,
                y=company_dist.index,
                orientation='h',
                marker_color=COLOR_PALETTE["primary"],
                hovertemplate='<b>Company:</b> %{y}<br><b>Total Amount:</b> %{x:,.2f} EGP<extra></extra>'
            )])
            fig_company.update_layout(
                title="Purchases by Company (Filtered)",
                xaxis_title="Amount (EGP)",
                yaxis_title="Company",
                height=400
            )
            st.plotly_chart(fig_company, use_container_width=True)
        else:
            st.info("No data matching filters to display company distribution.")

    with analysis_cols[1]:
        st.markdown(add_tooltip("Monthly Purchase Trend (Filtered)", 
                                "Trend of total purchase amounts per month for the filtered results."), 
                    unsafe_allow_html=True)
        if not search_results.empty:
            # Ensure 'date' column is datetime
            search_results['date'] = pd.to_datetime(search_results['date'])
            monthly_purchases = search_results.groupby(
                search_results["date"].dt.to_period("M")
            )["Invoice Amount"].sum()
            
            # Convert PeriodIndex to string for plotting if needed, or handle directly if Plotly supports it
            monthly_purchases.index = monthly_purchases.index.astype(str) 

            fig_monthly = go.Figure(data=[go.Scatter(
                x=monthly_purchases.index,
                y=monthly_purchases.values,
                mode='lines+markers',
                line=dict(color=COLOR_PALETTE["secondary"]),
                hovertemplate='<b>Month:</b> %{x}<br><b>Total Amount:</b> %{y:,.2f} EGP<extra></extra>'
            )])
            fig_monthly.update_layout(
                title="Monthly Purchase Trend (Filtered)",
                xaxis_title="Month",
                yaxis_title="Amount (EGP)",
                height=400
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
        else:
            st.info("No data matching filters to display monthly trend.")

    # Export functionality
    st.markdown(add_tooltip("📥 Export Results", 
                            "Download the filtered search results or a summary report."), 
                unsafe_allow_html=True)
    export_cols = st.columns(2)
    
    with export_cols[0]:
        excel_button_disabled = search_results.empty
        if st.button("Export Filtered Results to Excel", disabled=excel_button_disabled, 
                     help="Download the currently displayed filtered purchase records as an Excel file." if not excel_button_disabled else "No filtered results to export."):
            if not search_results.empty:
                with st.spinner("Preparing Excel export..."):
                    show_loading_spinner()
                    # Create Excel writer
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        # Write search results
                        search_results.to_excel(writer, sheet_name='Search Results', index=False)
                        
                        # Write summary based on filtered results
                        summary_data = pd.DataFrame({
                            'Metric': ['Filtered Purchases', 'Filtered Amount', 'Filtered Average', 'Unique Companies (Filtered)'],
                            'Value': [filtered_total_purchases, filtered_total_amount, filtered_avg_purchase, search_results["Invoice Company"].nunique()]
                        })
                        summary_data.to_excel(writer, sheet_name='Summary', index=False)
                    
                    # Create download button
                    output.seek(0)
                    st.download_button(
                        label="📥 Download Excel Report",
                        data=output,
                        file_name=f"inventory_purchase_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="download_excel_search" # Add unique key
                    )
            else:
                st.warning("No filtered results to export.") # Should not happen if button is disabled, but good practice
    
    with export_cols[1]:
        # PDF generation is complex, keeping it simple
        st.button("Generate PDF Report (Coming Soon)", disabled=True, 
                  help="Feature to generate a PDF report of the filtered results (currently unavailable).")
        # if st.button("Generate PDF Report"):
        #     if not search_results.empty:
        #         with st.spinner("Preparing PDF report..."):
        #             show_loading_spinner()
        #             st.info("PDF report generation will be implemented in the next version.")
        #     else:
        #         st.warning("No filtered results to generate a PDF report.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Pharmacy Analytics Dashboard • Updated: "
    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>", 
    unsafe_allow_html=True
)
