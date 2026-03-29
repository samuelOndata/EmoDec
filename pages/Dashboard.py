import os
import streamlit as st
import pandas as pd
import plotly.express as px
from db.db_utils import get_all_predictions
from dotenv import load_dotenv

load_dotenv()

# --- ACCESS CONTROL ---
ACCESS_CODE = os.getenv("DASHBOARD_ACCESS_CODE", "admin123") # Fallback to admin123

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Admin Access Required")
    
    # Use type="password" to hide the characters as they are typed
    user_input = st.text_input("Enter Access Code:", type="password")
    
    if st.button("Login"):
        if user_input == ACCESS_CODE:
            st.session_state.authenticated = True
            st.rerun() # Refresh to show the dashboard
        else:
            st.error("Invalid code. Access denied.")
    st.stop() # Stops execution here so the rest of the code doesn't run

# --- EVERYTHING BELOW RUNS ONLY IF AUTHENTICATED IS TRUE ---

st.title("📊 Model Analytics Dashboard")

# Add a logout button in the sidebar
if st.sidebar.button("Log Out"):
    st.session_state.authenticated = False
    st.rerun()

# 1. Fetch data from your SQL table
data = get_all_predictions()

if data:
    # 2. Create DataFrame with your specific SQL column names
    df = pd.DataFrame(data, columns=[
        'id', 'image_url', 'predicted_label', 'correct_label', 
        'confidence', 'is_correct', 'created_at'
    ])

    # --- CALCULATIONS ---
    total_predictions = len(df)
    total_images = df['image_url'].nunique()  # Number of unique images uploaded
    
    # Accept/Reject Rates
    # Since you have an 'is_correct' boolean, we use that:
    accepts = df['is_correct'].sum() 
    rejects = total_predictions - accepts
    
    accept_rate = (accepts / total_predictions) * 100 if total_predictions > 0 else 0
    reject_rate = (rejects / total_predictions) * 100 if total_predictions > 0 else 0

    # --- KPI SECTION ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Predictions", total_predictions)
    col2.metric("Unique Images", total_images)
    col3.metric("Accept Rate ✅", f"{accept_rate:.1f}%")
    col4.metric("Reject Rate ❌", f"{reject_rate:.1f}%", delta=f"{reject_rate:.1f}%", delta_color="inverse")

    st.divider()

    # --- VISUALIZATION SECTION ---
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Model Accuracy per Emotion")
        # Group by the predicted label and calculate the mean of 'is_correct'
        # Multiply by 100 to get a percentage
        accuracy_df = df.groupby('correct_label')['is_correct'].mean().reset_index()
        accuracy_df['is_correct'] *= 100
        
        fig_acc = px.bar(
            accuracy_df, 
            x='correct_label', 
            y='is_correct',
            labels={'is_correct': 'Accuracy %', 'correct_label': 'Actual Emotion'},
            color='is_correct',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_acc, width='stretch')

    with chart_col2:
        st.subheader("Prediction Distribution")
        emotion_counts = df['predicted_label'].value_counts().reset_index()
        
        fig_pie = px.pie(
            emotion_counts, 
            values='count', 
            names='predicted_label', 
            hole=0.4
        )
        st.plotly_chart(fig_pie, width='stretch')

    # --- RAW DATA TABLE ---
    with st.expander("📂 View Full Prediction History"):
        # Show recent first
        st.dataframe(
            df.sort_values(by='created_at', ascending=False), 
            width='stretch'
        )

else:
    st.info("No data found. Upload an image and submit feedback to populate the dashboard!")

# Simple refresh button to pull new SQL data
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()