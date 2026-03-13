import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np
import anthropic

# --- Setup ---
st.set_page_config(page_title="Sales Dashboard", page_icon="📊", layout="wide")
df = pd.read_csv("cleaned_superstore.csv")

# --- Sidebar Filters ---
st.sidebar.title("🔍 Filters")
years = sorted(df['Year'].unique())
selected_year = st.sidebar.selectbox("Select Year", ["All"] + years)
selected_region = st.sidebar.selectbox("Select Region", ["All"] + list(df['Region'].unique()))

if selected_year != "All":
    df = df[df['Year'] == selected_year]
if selected_region != "All":
    df = df[df['Region'] == selected_region]

# --- Page Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "🗺️ Regional", "📦 Categories", "🔮 Forecast", "🤖 Ask AI"])

# --- Tab 1: Overview ---
with tab1:
    st.title("📈 Sales Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
    col2.metric("Total Profit", f"${df['Profit'].sum():,.0f}")
    col3.metric("Total Orders", f"{df['Order ID'].nunique():,}")
    col4.metric("Avg Profit Margin", f"{df['Profit Margin'].mean()*100:.1f}%")

    st.subheader("Monthly Sales Trend")
    monthly = df.groupby("Month")["Sales"].sum().reset_index()
    fig = px.line(monthly, x="Month", y="Sales", markers=True, color_discrete_sequence=["#636EFA"])
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Regional ---
with tab2:
    st.title("🗺️ Regional Analysis")
    region_df = df.groupby("Region")[["Sales", "Profit"]].sum().reset_index()
    fig = px.bar(region_df, x="Region", y=["Sales", "Profit"], barmode="group",
                 color_discrete_sequence=["#636EFA", "#EF553B"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sales by State")
    state_df = df.groupby("State")["Sales"].sum().reset_index()
    fig2 = px.choropleth(state_df, locations="State", locationmode="USA-states",
                         color="Sales", scope="usa", color_continuous_scale="Blues")
    st.plotly_chart(fig2, use_container_width=True)

# --- Tab 3: Categories ---
with tab3:
    st.title("📦 Category & Product Analysis")
    cat_df = df.groupby("Category")[["Sales", "Profit"]].sum().reset_index()
    fig = px.pie(cat_df, names="Category", values="Sales", title="Sales by Category")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 Sub-Categories by Profit")
    sub_df = df.groupby("Sub-Category")["Profit"].sum().reset_index().sort_values("Profit", ascending=False).head(10)
    fig2 = px.bar(sub_df, x="Profit", y="Sub-Category", orientation="h",
                  color="Profit", color_continuous_scale="RdYlGn")
    st.plotly_chart(fig2, use_container_width=True)

# --- Tab 4: Forecast ---
with tab4:
    st.title("🔮 Revenue Forecast (Next 3 Months)")
    monthly_all = pd.read_csv("cleaned_superstore.csv").groupby("Month")["Sales"].sum().reset_index()
    monthly_all["Month_Num"] = range(len(monthly_all))

    X = monthly_all[["Month_Num"]]
    y = monthly_all["Sales"]
    model = LinearRegression().fit(X, y)

    future_nums = [len(monthly_all), len(monthly_all)+1, len(monthly_all)+2]
    future_labels = ["Month +1", "Month +2", "Month +3"]
    predictions = model.predict(pd.DataFrame({"Month_Num": future_nums}))

    forecast_df = pd.DataFrame({"Month": future_labels, "Predicted Sales": predictions.round(0)})

    combined = pd.concat([
        monthly_all[["Month", "Sales"]].rename(columns={"Sales": "Predicted Sales"}),
        forecast_df
    ])
    combined["Type"] = ["Actual"] * len(monthly_all) + ["Forecast"] * 3

    fig = px.line(combined, x="Month", y="Predicted Sales", color="Type",
                  color_discrete_map={"Actual": "#636EFA", "Forecast": "#FF7F0E"},
                  markers=True)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(forecast_df, use_container_width=True)

# --- Tab 5: Ask AI ---
with tab5:
    st.title("🤖 Ask AI About Your Data")
    st.caption("Ask anything about the sales data — powered by Claude AI")

    api_key = st.text_input("Enter your Claude API Key", type="password")
    question = st.text_input("Ask a question", placeholder="e.g. Which region has the lowest profit margin?")

    if st.button("Ask") and api_key and question:
        summary = f"""
        Total Sales: ${df['Sales'].sum():,.0f}
        Total Profit: ${df['Profit'].sum():,.0f}
        Regions: {df.groupby('Region')['Sales'].sum().to_dict()}
        Categories: {df.groupby('Category')['Sales'].sum().to_dict()}
        Avg Profit Margin: {df['Profit Margin'].mean()*100:.1f}%
        Top States by Sales: {df.groupby('State')['Sales'].sum().nlargest(5).to_dict()}
        """
        with st.spinner("Thinking..."):
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": f"Sales data summary:\n{summary}\n\nQuestion: {question}"}]
            )
            st.success(response.content[0].text)