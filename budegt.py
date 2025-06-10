import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

st.set_page_config(layout="wide")
st.title("📊 Budget & Procurement Analytics with AI")

@st.cache_data
def load_all():
    budget_df = pd.read_csv("final_budget_data.csv")
    plan_df = pd.read_csv("procurement_plan_500_records_with_planned_and_spent.csv")
    spent_df = pd.read_csv("actual_procurement_spent_500.csv")
    return budget_df, plan_df, spent_df

budget_df, plan_df, spent_df = load_all()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📘 Final Budget", 
    "📝 Plan View", 
    "💵 Actual Spend", 
    "📈 Predictive Analysis", 
    "🤖 AI Prompt Analysis"
])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("📘 Final Budget Overview")
    st.dataframe(budget_df)
    fig1 = px.bar(budget_df, x="MONTH_NAME", y="ACCOUNTED_DR", color="budegt_CHAPTER", title="Monthly Budget Spend by Chapter")
    st.plotly_chart(fig1, use_container_width=True)

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("📝 Procurement Plan Details")
    st.dataframe(plan_df)
    fig2 = px.bar(plan_df, x="Department", y="Total AED", color="Purchase Type", title="Planned Procurement by Department")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("💵 Actual Procurement Spend")
    st.dataframe(spent_df)
    fig3 = px.bar(spent_df, x="Department", y="Actual Spent", color="Purchase Type", title="Actual Spend by Department")
    st.plotly_chart(fig3, use_container_width=True)

# ---------------- TAB 4 ----------------
with tab4:
    st.subheader("📈 Forecasting Actual Spend")

    forecast_basis = st.selectbox("Forecast By:", ["Department", "Account No."])
    group_col = "Department" if forecast_basis == "Department" else "Account No."

    selected_group = st.selectbox(f"Select {group_col}", spent_df[group_col].unique())
    subset = spent_df[spent_df[group_col] == selected_group]

    # Create time series (Month -> Spend)
    monthly_columns = [col for col in subset.columns if col.endswith("-24")]
    ts_df = subset[monthly_columns].sum().reset_index()
    ts_df.columns = ["ds", "y"]
    ts_df["ds"] = pd.to_datetime(ts_df["ds"], format="%b-%y")

    # Forecast
    model = Prophet()
    model.fit(ts_df)
    future = model.make_future_dataframe(periods=3, freq='M')
    forecast = model.predict(future)

    fig_forecast = px.line(forecast, x="ds", y="yhat", title=f"Forecasted Spend - {selected_group}")
    st.plotly_chart(fig_forecast, use_container_width=True)

# ---------------- TAB 5 ----------------
with tab5:
    st.subheader("🤖 LLM-Based Prompt to Graph")

    user_prompt = st.text_area("Enter prompt:", "Show bar chart of Actual Spent by Department")

    if st.button("Generate Chart"):
        llm = ChatOpenAI(temperature=0, model="gpt-4")
        agent = create_pandas_dataframe_agent(llm, spent_df, agent_type=AgentType.OPENAI_FUNCTIONS, verbose=True)
        try:
            result = agent.run(user_prompt)
            st.success("LLM Response:")
            st.write(result)
        except Exception as e:
            st.error(f"LLM Error: {e}")
