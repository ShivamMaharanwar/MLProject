import streamlit as st
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# 1. Page Configuration
st.set_page_config(
    page_title="Student Exam Performance Indicator",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 2. Injecting Custom Vibrant CSS Styling (Dark Cyberpunk Glassmorphic Theme)
st.markdown("""
    <style>
        /* Main background layout */
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            color: #f8fafc;
        }
        
        /* Container styling */
        .main-card {
            background: rgba(255, 255, 255, 0.06);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 2.5rem;
            border-radius: 24px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
            margin-bottom: 2rem;
        }
        
        /* Gradient typography headings */
        .main-title {
            font-size: 2.5rem !construct;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0.25rem;
            background: linear-gradient(to right, #ffffff, #c084fc, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .sub-title {
            font-size: 1rem;
            color: #94a3b8;
            text-align: center;
            margin-bottom: 2.5rem;
        }

        /* Customize selectbox labels & number inputs */
        label {
            color: #cbd5e1 !important;
            font-weight: 500 !important;
            font-size: 0.9rem !important;
        }

        /* Success banner customized styling */
        .result-box {
            margin-top: 2rem;
            padding: 1.5rem;
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.15) 100%);
            border: 1px solid rgba(16, 185, 129, 0.3);
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        .result-box h2 {
            font-size: 1.2rem !important;
            color: #34d399 !important;
            margin: 0 !important;
            padding: 0 !important;
        }
        .result-box .score {
            font-size: 2.2rem !important;
            font-weight: 800 !important;
            color: #10b981 !important;
            display: block;
            margin-top: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# 3. App Header Render
st.markdown('<h1 class="main-title">Student Exam Performance</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Predict Student\'s Mathematics Performance Indicator</p>', unsafe_allow_html=True)

# 4. Form Container
with st.container():
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    
    # Using columns to create a balanced form grid structure
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["male", "female"], index=None, placeholder="Select Gender")
        
        parental_level_of_education = st.selectbox(
            "Parental Level of Education",
            [
                "associate's degree",
                "bachelor's degree",
                "high school",
                "master's degree",
                "some college",
                "some high school"
            ],
            index=None,
            placeholder="Select Parent Education"
        )
        
        test_preparation_course = st.selectbox("Test Preparation Course", ["none", "completed"], index=None, placeholder="Select Status")

    with col2:
        ethnicity = st.selectbox("Race or Ethnicity", ["group A", "group B", "group C", "group D", "group E"], index=None, placeholder="Select Ethnicity")
        
        lunch = st.selectbox("Lunch Type", ["free/reduced", "standard"], index=None, placeholder="Select Lunch Type")

    # Score fields split nicely at the bottom
    col3, col4 = st.columns(2)
    with col3:
        reading_score = st.number_input("Reading Score (0-100)", min_value=0, max_value=100, value=None, placeholder="Enter Score")
    with col4:
        writing_score = st.number_input("Writing Score (0-100)", min_value=0, max_value=100, value=None, placeholder="Enter Score")

    st.markdown('<br>', unsafe_allow_html=True)
    
    # 5. Prediction Execution
    if st.button("Predict Maths Score", use_container_width=True, type="primary"):
        # Explicit validation check to ensure user filled all input slots
        if None in [gender, ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score]:
            st.error("⚠️ Please fill out all configuration and score inputs before predicting.")
        else:
            with st.spinner("Processing framework pipeline prediction..."):
                try:
                    # Construct data object mapping fields exactly to custom class structure
                    data = CustomData(
                        gender=gender,
                        race_ethnicity=ethnicity,
                        parental_level_of_education=parental_level_of_education,
                        lunch=lunch,
                        test_preparation_course=test_preparation_course,
                        reading_score=float(reading_score),
                        writing_score=float(writing_score)
                    )
                    
                    pred_df = data.get_data_as_data_frame()
                    
                    predict_pipeline = PredictPipeline()
                    results = predict_pipeline.predict(pred_df)
                    rounded_result = round(results[0], 2)
                    
                    # Custom UI Container Render for the Output Score
                    st.markdown(f"""
                        <div class="result-box">
                            <h2>Predicted Maths Score</h2>
                            <span class="score">{rounded_result}</span>
                        </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"An error occurred in the prediction pipeline: {e}")
                    
    st.markdown('</div>', unsafe_allow_html=True)
