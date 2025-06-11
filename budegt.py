import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

# LangChain / OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# UI Config
st.set_page_config(layout="wide")
st.title("📊 Budget & Procurement Dashboard with AI Insights")

@st.cache_data
def load_data():
    budget_df = pd.read_csv("final_budget_data.csv")
    plan_df = pd.read_csv("procurement_plan_500_records_with_planned_and_spent.csv")
    spent_df = pd.read_csv("actual_procurement_spent_500.csv")
    return budget_df, plan_df, spent_df

budget_df, plan_df, spent_df = load_data()

# --- API Key Input ---
st.sidebar.subheader("🔐 OpenAI API Access")
user_api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📘 Final Budget", 
    "📝 Procurement Plan", 
    "💵 Actual Spend", 
    "📈 Forecasting", 
    "🤖 AI Prompt Insights"
])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("📘 Final Budget Overview")
    st.dataframe(budget_df)
    fig1 = px.bar(
        budget_df, x="MONTH_NAME", y="ACCOUNTED_DR", color="budegt_CHAPTER",
        title="Monthly Budget Spend by Chapter"
    )
    st.plotly_chart(fig1, use_container_width=True)

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("📝 Procurement Plan Details")
    st.dataframe(plan_df)
    fig2 = px.bar(
        plan_df, x="Department", y="Total AED", color="Purchase Type",
        title="Planned Procurement by Department"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("💵 Actual Procurement Spend")
    st.dataframe(spent_df)
    fig3 = px.bar(
        spent_df, x="Department", y="Actual Spent", color="Purchase Type",
        title="Actual Spend by Department"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ---------------- TAB 4 ----------------
with tab4:
    st.subheader("📈 Forecasting Spend")

    forecast_by = st.selectbox("Forecast By", ["Department", "Account No."])
    group = forecast_by
    selected = st.selectbox(f"Select {group}", spent_df[group].unique())

    subset = spent_df[spent_df[group] == selected]
    monthly_cols = [col for col in subset.columns if col.endswith("-24")]
    
    if not monthly_cols:
        st.warning("No monthly columns found for forecasting.")
    else:
        ts_df = subset[monthly_cols].sum().reset_index()
        ts_df.columns = ["ds", "y"]
        ts_df["ds"] = pd.to_datetime(ts_df["ds"], format="%b-%y")

        model = Prophet()
        model.fit(ts_df)
        future = model.make_future_dataframe(periods=3, freq='M')
        forecast = model.predict(future)

        fig4 = px.line(forecast, x="ds", y="yhat", title=f"Forecast: {forecast_by} - {selected}")
        st.plotly_chart(fig4, use_container_width=True)

# ---------------- TAB 5 ----------------
with tab5:
    st.subheader("🤖 AI Chart Generator from Prompt")

    if not user_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to use this feature.")
    else:
        prompt = st.text_area("Ask something (e.g., 'Show bar chart of Actual Spent by Department'):")

        if st.button("Generate Chart from Prompt") and prompt:
            try:
                llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=user_api_key)
                agent = create_pandas_dataframe_agent(llm, spent_df, agent_type=AgentType.OPENAI_FUNCTIONS, verbose=True)
                result = agent.run(prompt)
                st.success("AI Response:")
                st.write(result)
            except Exception as e:
                st.error(f"Failed to process AI chart generation:\n{e}")
