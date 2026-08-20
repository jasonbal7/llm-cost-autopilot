import os
import streamlit as st
import pandas as pd
import plotly.express as px

# Configure layout
st.set_page_config(page_title="LLM Cost Autopilot", layout="wide")
st.title("LLM Cost Autopilot Telemetry")

AUDIT_LOG = "data/master_audit_log.csv"
ESC_LOG = "data/escalation_log.csv"


if not os.path.exists(AUDIT_LOG):
    st.warning("No audit log found. Run main.py to generate traffic.")
else:
    df_audit = pd.read_csv(AUDIT_LOG)
    
    # 1. Calculate Core Financials
    total_reqs = len(df_audit)
    actual_cost = df_audit["actual_cost"].sum()
    max_cost = df_audit["hypothetical_max_cost"].sum()
    savings = df_audit["savings_usd"].sum()
    
    # 2. The Money Shot Metric
    savings_pct = (savings / max_cost * 100) if max_cost > 0 else 0
    
    st.markdown(f"### **Total System Savings: {savings_pct:.1f}%**")
    
    # Top Level Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Requests", total_reqs)
    c2.metric("Actual Cost", f"${actual_cost:.4f}")
    c3.metric("Cost w/o Router", f"${max_cost:.4f}")
    c4.metric("Money Saved", f"${savings:.4f}")
    
    st.divider()

    # 3. Visualizations
    col_pie, col_bar = st.columns(2)
    
    with col_pie:
        st.subheader("Routing Distribution")
        fig_pie = px.pie(df_audit, names="model_used", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_bar:
        st.subheader("Escalation Rate")
        esc_counts = df_audit["escalated"].value_counts().reset_index()
        esc_counts.columns = ["Escalated", "Count"]
        fig_bar = px.bar(esc_counts, x="Escalated", y="Count", color="Escalated", color_discrete_sequence=["#2ca02c", "#d62728"])
        st.plotly_chart(fig_bar, use_container_width=True)

    # 4. Quality Score Distribution
    if os.path.exists(ESC_LOG):
        st.divider()
        st.subheader("Quality Verifier Scores")
        df_esc = pd.read_csv(ESC_LOG)
        fig_hist = px.histogram(df_esc, x="score", nbins=10, color="passed")
        st.plotly_chart(fig_hist, use_container_width=True)