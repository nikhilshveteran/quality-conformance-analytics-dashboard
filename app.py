import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Quality Conformance Analytics", layout="wide")
st.title("🚀 Quality Conformance Analytics Dashboard")
st.caption("Excellence through Data-Driven Decision Making | Created by Nikhil Sharma")

# Load Data
df = pd.read_csv("quality_conformance_data.csv")
df['Production_Date'] = pd.to_datetime(df['Production_Date'])

# Sidebar Filters
st.sidebar.header("Filter Data")
start_date = st.sidebar.date_input("Start Date", df['Production_Date'].min())
end_date = st.sidebar.date_input("End Date", df['Production_Date'].max())
supplier_filter = st.sidebar.multiselect("Supplier", df['Supplier_ID'].unique(), df['Supplier_ID'].unique())

filtered_df = df[(df['Production_Date'] >= pd.to_datetime(start_date)) &
                 (df['Production_Date'] <= pd.to_datetime(end_date)) &
                 (df['Supplier_ID'].isin(supplier_filter))]

# KPI Cards
total_products = filtered_df.shape[0]
pass_count = filtered_df[filtered_df['Inspection_Result'] == 'Pass'].shape[0]
fail_count = filtered_df[filtered_df['Inspection_Result'] == 'Fail'].shape[0]
rework_count = filtered_df[filtered_df['Rework_Required'] == 'Yes'].shape[0]
total_defect_cost = filtered_df['Cost_of_Defect'].sum()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Products", total_products)
col2.metric("Pass Count", pass_count)
col3.metric("Fail Count", fail_count)
col4.metric("Rework Count", rework_count)
col5.metric("Total Defect Cost (€)", f"{total_defect_cost:,.2f}")

# 📢 Anomaly Detection - Z-Score Method
st.subheader("⚡ Anomaly Detection: Defect Spike Alert")

defects_daily = filtered_df.groupby('Production_Date').size().reset_index(name='Defect_Count')
defects_daily['Z_Score'] = (defects_daily['Defect_Count'] - defects_daily['Defect_Count'].mean()) / defects_daily['Defect_Count'].std()

anomalies = defects_daily[abs(defects_daily['Z_Score']) > 2]

if not anomalies.empty:
    st.warning("🚨 Anomalies Detected in Defect Data:")
    st.dataframe(anomalies[['Production_Date', 'Defect_Count', 'Z_Score']])
else:
    st.success("✅ No significant anomalies detected in the current data range.")

# 🏆 Supplier Performance Benchmarking
st.subheader("🏅 Supplier Performance Benchmarking")

supplier_quality = filtered_df.groupby('Supplier_ID')['Inspection_Result'].value_counts(normalize=True).unstack().fillna(0)
supplier_quality['Defect Rate'] = supplier_quality.get('Fail', 0)

benchmark_table = supplier_quality.sort_values('Defect Rate', ascending=True).reset_index()
st.dataframe(benchmark_table[['Supplier_ID', 'Defect Rate']].rename(columns={'Defect Rate': 'Defect Rate (%)'}))

top_supplier = benchmark_table.iloc[0]['Supplier_ID']
worst_supplier = benchmark_table.iloc[-1]['Supplier_ID']

st.info(f"🌟 **Top Performing Supplier:** {top_supplier}")
st.error(f"⚠️ **Supplier Needing Attention:** {worst_supplier}")

# 💾 Export Filtered Data Section
st.subheader("💾 Export Filtered Report")

csv = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv,
    file_name='quality_conformance_filtered_data.csv',
    mime='text/csv',
)

st.info("✅ You can use this export for audits or offline analysis.")



# Six Sigma Control Chart
defects_per_day = filtered_df.groupby('Production_Date').size().reset_index(name='Defect_Count')
mean = defects_per_day['Defect_Count'].mean()
std_dev = defects_per_day['Defect_Count'].std()

defects_per_day['UCL'] = mean + 3 * std_dev
defects_per_day['LCL'] = mean - 3 * std_dev
defects_per_day['Mean'] = mean

fig6 = px.line(defects_per_day, x='Production_Date', y='Defect_Count', title='Six Sigma Control Chart')
fig6.add_scatter(x=defects_per_day['Production_Date'], y=defects_per_day['UCL'], mode='lines', name='UCL (+3σ)')
fig6.add_scatter(x=defects_per_day['Production_Date'], y=defects_per_day['LCL'], mode='lines', name='LCL (-3σ)')
fig6.add_scatter(x=defects_per_day['Production_Date'], y=defects_per_day['Mean'], mode='lines', name='Mean (µ)')
st.plotly_chart(fig6, use_container_width=True)

# 💡 Cost Impact Forecast using Linear Regression
from sklearn.linear_model import LinearRegression

st.subheader("💸 Cost Impact Forecast")

# Prepare data for modeling
cost_df = filtered_df.groupby(filtered_df['Production_Date'].dt.to_period('M'))['Cost_of_Defect'].sum().reset_index()
cost_df['Production_Date'] = pd.to_datetime(cost_df['Production_Date'].astype(str))
cost_df['Months'] = (cost_df['Production_Date'] - cost_df['Production_Date'].min()).dt.days

if len(cost_df) > 3:
    X = cost_df[['Months']]
    y = cost_df['Cost_of_Defect']
    model = LinearRegression().fit(X, y)

    # Predict next 3 months
    future_months = pd.DataFrame({'Months': [X['Months'].max() + 30 * i for i in range(1, 4)]})
    future_costs = model.predict(future_months)

    for i, cost in enumerate(future_costs):
        st.write(f"📅 Month {i+1}: Predicted Defect Cost = €{cost:,.2f}")
else:
    st.info("Need more historical data for accurate forecasting.")


# 🔥 Root Cause vs Shift Heatmap
st.subheader("🔥 Root Cause - Shift Heatmap")

heatmap_data = filtered_df[filtered_df['Root_Cause_Category'] != 'N/A']
pivot_table = pd.pivot_table(
    heatmap_data, 
    values='Product_ID', 
    index='Root_Cause_Category', 
    columns='Shift', 
    aggfunc='count',
    fill_value=0
)

st.write("Root Cause occurrence count per Shift:")
st.dataframe(pivot_table.style.background_gradient(cmap='Reds'))


# Pareto Chart
defect_counts = filtered_df['Defect_Type'].value_counts().reset_index()
defect_counts.columns = ['Defect Type', 'Count']
defect_counts['Cumulative %'] = defect_counts['Count'].cumsum() / defect_counts['Count'].sum() * 100

fig7 = px.bar(defect_counts, x='Defect Type', y='Count', title='Pareto Chart - Defect Types')
fig7.add_scatter(x=defect_counts['Defect Type'], y=defect_counts['Cumulative %'], mode='lines+markers', name='Cumulative %')
st.plotly_chart(fig7, use_container_width=True)

# Sigma Level Calculator
if fail_count > 0:
    defects_per_million = (fail_count / total_products) * 1_000_000
    sigma_level = 1.5 + 0.8406 * np.log10((total_products - fail_count) / fail_count) if fail_count else 6
else:
    sigma_level = 6.0

st.subheader("📏 Sigma Level Estimate")
st.write(f"**Defects Per Million (DPMO):** {defects_per_million:.2f}")
st.write(f"**Estimated Sigma Level:** {sigma_level:.2f} σ")

# Process Capability (Cp, Cpk) Calculation (mock spec limits for illustration)
st.subheader("📊 Process Capability (Cp & Cpk)")
spec_lower = st.number_input("Enter Lower Specification Limit", value=0)
spec_upper = st.number_input("Enter Upper Specification Limit", value=50)

if not filtered_df.empty:
    process_mean = defects_per_day['Defect_Count'].mean()
    process_std = defects_per_day['Defect_Count'].std()
    cp = (spec_upper - spec_lower) / (6 * process_std) if process_std > 0 else 0
    cpk = min((spec_upper - process_mean) / (3 * process_std),
              (process_mean - spec_lower) / (3 * process_std)) if process_std > 0 else 0
    st.write(f"**Cp:** {cp:.2f}")
    st.write(f"**Cpk:** {cpk:.2f}")
else:
    st.info("No data available for Cp & Cpk calculation in the selected range.")

# Predictive Analytics - Defect Prediction Model
st.subheader("🤖 Predictive Analytics: Defect Prediction")

# Simplified binary model: Rework Required based on numeric fields
model_df = pd.get_dummies(filtered_df[['Cost_of_Defect', 'Rework_Required']], drop_first=True)
if len(model_df) > 10:
    X = model_df[['Cost_of_Defect']]
    y = model_df['Rework_Required_Yes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Prediction Model Accuracy: **{accuracy:.2%}**")
    st.write("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
else:
    st.warning("Not enough data to train a predictive model — try a wider date range or full dataset.")

st.markdown("---")
st.caption("© 2025 | Quality Conformance Dashboard — Created by Nikhil Sharma")



