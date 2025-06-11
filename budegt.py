import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
import re
import io
import contextlib

from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

st.set_page_config(layout="wide")
st.title("\U0001F4CA Budget & Procurement Dashboard with AI Insights")

@st.cache_data
def load_data():
    budget_df = pd.read_csv("final_budget_data.csv")
    plan_df = pd.read_csv("procurement_plan_500_records_with_planned_and_spent.csv")
    spent_df = pd.read_csv("actual_procurement_spent_500.csv")
    return budget_df, plan_df, spent_df

budget_df, plan_df, spent_df = load_data()

st.sidebar.subheader("\U0001F512 OpenAI API Access")
user_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "\U0001F4D8 Final Budget Insights",
    "\U0001F4DD Plan Analysis",
    "\U0001F4B5 Actual Spend Trends",
    "\U0001F4C8 Forecast Center",
    "\U0001F916 AI Chat Insights"
])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Top Budget Chapters by Spend")
    top_chapters = budget_df.groupby("budegt_CHAPTER")["ACCOUNTED_DR"].sum().sort_values(ascending=False).head(5).reset_index()
    st.plotly_chart(px.bar(top_chapters, x="budegt_CHAPTER", y="ACCOUNTED_DR", title="Top 5 Budget Chapters by Spend"))

    st.subheader("Monthly Spend Trend (Forecast)")
    monthly = budget_df.groupby("MONTH_NAME")["ACCOUNTED_DR"].sum().reset_index()
    monthly["ds"] = pd.to_datetime("2024-" + monthly["MONTH_NAME"], format="%Y-%b")
    monthly = monthly.sort_values("ds")
    monthly.rename(columns={"ACCOUNTED_DR": "y"}, inplace=True)
    model = Prophet()
    model.fit(monthly[["ds", "y"]])
    future = model.make_future_dataframe(periods=3, freq='M')
    forecast = model.predict(future)
    st.plotly_chart(px.line(forecast, x="ds", y="yhat", title="Forecasted Budget Spend"))

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("Department-Wise Planned Procurement")
    dept_totals = plan_df.groupby("Department")["Total AED"].sum().sort_values(ascending=False).reset_index()
    st.plotly_chart(px.bar(dept_totals, x="Department", y="Total AED", title="Planned Procurement per Department"))

    st.subheader("Monthly Procurement Plan Trend")
    monthly_cols = [col for col in plan_df.columns if col.endswith("-24")]
    trend_df = plan_df[monthly_cols].sum().reset_index()
    trend_df.columns = ["Month", "Spend"]
    trend_df["ds"] = pd.to_datetime(trend_df["Month"], format="%b-%y")
    trend_df = trend_df.sort_values("ds")
    model = Prophet()
    model.fit(trend_df[["ds", "Spend"]].rename(columns={"Spend": "y"}))
    future = model.make_future_dataframe(periods=3, freq="M")
    forecast = model.predict(future)
    st.plotly_chart(px.line(forecast, x="ds", y="yhat", title="Forecasted Procurement Plan Spend"))

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("Top Departments by Actual Spend")
    top_depts = spent_df.groupby("Department")["Actual Spent"].sum().sort_values(ascending=False).head(5).reset_index()
    st.plotly_chart(px.bar(top_depts, x="Department", y="Actual Spent", title="Top 5 Departments by Actual Spend"))

    st.subheader("Monthly Actual Spend Forecast")
    monthly_cols = [col for col in spent_df.columns if col.endswith("-24")]
    actual_monthly = spent_df[monthly_cols].sum().reset_index()
    actual_monthly.columns = ["Month", "Spend"]
    actual_monthly["ds"] = pd.to_datetime(actual_monthly["Month"], format="%b-%y")
    actual_monthly = actual_monthly.sort_values("ds")
    model = Prophet()
    model.fit(actual_monthly[["ds", "Spend"]].rename(columns={"Spend": "y"}))
    future = model.make_future_dataframe(periods=3, freq="M")
    forecast = model.predict(future)
    st.plotly_chart(px.line(forecast, x="ds", y="yhat", title="Forecasted Actual Spend"))

# ---------------- TAB 4 ----------------
with tab4:
    st.subheader("Select Forecast Entity")
    target_col = st.selectbox("Forecast By", ["Department", "Account No.", "PERIOD_NAME"])
    selected_value = st.selectbox(f"Select {target_col}", spent_df[target_col].dropna().unique())
    monthly_cols = [col for col in spent_df.columns if col.endswith("-24")]
    subset = spent_df[spent_df[target_col] == selected_value]
    ts_df = subset[monthly_cols].sum().reset_index()
    ts_df.columns = ["ds", "y"]
    ts_df["ds"] = pd.to_datetime(ts_df["ds"], format="%b-%y")
    model = Prophet()
    model.fit(ts_df)
    future = model.make_future_dataframe(periods=3, freq="M")
    forecast = model.predict(future)
    st.plotly_chart(px.line(forecast, x="ds", y="yhat", title=f"Forecast for {target_col}: {selected_value}"))

# ---------------- TAB 5 ----------------
with tab5:
    st.subheader("\U0001F916 AI Chart Generator from Prompt")
    if not user_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to use this feature.")
    else:
        prompt = st.text_area("Ask anything about actual spend data:")
        if st.button("Generate Insight") and prompt:
            try:
                llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=user_api_key)
                agent = create_pandas_dataframe_agent(
                    llm, spent_df,
                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    allow_dangerous_code=True,
                    verbose=True,
                    handle_parsing_errors=True
                )
                code_prompt = (
                    f"You are a Streamlit + Plotly Python analyst. "
                    f"Given the following user query, write Python code to generate the chart using dataframe `df` with Streamlit.\n\n"
                    f"Only output the code inside triple backticks.\n\n"
                    f"User Query: {prompt}"
                )
                result = agent.run(code_prompt)

                match = re.search(r"```(?:python)?\\n(.*?)```", result, re.DOTALL)
                if match:
                    code = match.group(1)
                    st.code(code, language="python")
                    with st.spinner("Generating chart..."):
                        with contextlib.redirect_stdout(io.StringIO()) as f:
                            exec_globals = {"df": spent_df, "pd": pd, "px": px, "st": st}
                            exec(code, exec_globals)
                        output = f.getvalue()
                        if output:
                            st.text(output)
                else:
                    st.warning("No Python code found. Displaying raw output:")
                    st.write(result)

            except Exception as e:
                st.error(f"Failed to process AI insight:\n{e}")
