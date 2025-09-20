# app.py
"""
Complete Sleep Recommendation Streamlit App
Make sure to run train_sleep_model.py first to generate sleep_model.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Sleep Recommendation System",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .recommendation-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained model package"""
    try:
        with open('sleep_model.pkl', 'rb') as f:
            model_package = pickle.load(f)
        return model_package
    except FileNotFoundError:
        st.error("❌ Model file 'sleep_model.pkl' not found!")
        st.info("Please run 'train_sleep_model.py' first to train and save the model.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_sleep_recommendations(user_data, model_package):
    """Make predictions using the trained models"""
    try:
        # Prepare input data
        input_df = pd.DataFrame([user_data])
        
        # Scale the input data
        input_scaled = model_package['scaler'].transform(input_df)
        
        # Make predictions
        quality_pred = model_package['quality_model'].predict(input_scaled)[0]
        duration_pred = model_package['duration_model'].predict(input_scaled)[0]
        bedtime_pred = model_package['bedtime_model'].predict(input_scaled)[0]
        
        return {
            'sleep_quality': max(1, min(10, quality_pred)),
            'sleep_duration': max(6, min(9, duration_pred)),
            'bedtime_hour': max(20, min(24, bedtime_pred))
        }
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def generate_recommendations(user_data, predictions):
    """Generate personalized recommendations"""
    recommendations = []
    
    # Caffeine recommendations
    if user_data['caffeine_intake'] > 300:
        recommendations.append("☕ Reduce caffeine intake to under 300mg/day")
    elif user_data['caffeine_intake'] > 200:
        recommendations.append("☕ Avoid caffeine after 2 PM")
    
    # Screen time recommendations
    if user_data['screen_time'] > 10:
        recommendations.append("📱 Reduce screen time, especially 2 hours before bed")
    elif user_data['screen_time'] > 8:
        recommendations.append("📱 Use blue light filters in the evening")
    
    # Stress recommendations
    if user_data['stress_level'] > 7:
        recommendations.append("🧘 Practice meditation or relaxation techniques")
        recommendations.append("📚 Try reading instead of screens before bed")
    
    # Activity recommendations
    if user_data['activity_level'] < 2:
        recommendations.append("🏃 Increase physical activity during the day")
        recommendations.append("🚶 Take a short walk after dinner")
    
    # BMI recommendations
    if user_data['bmi'] > 25:
        recommendations.append("⚖️ Consider weight management for better sleep quality")
    
    # Temperature recommendations
    if user_data['room_temperature'] > 22:
        recommendations.append("❄️ Keep bedroom temperature between 18-20°C")
    elif user_data['room_temperature'] < 18:
        recommendations.append("🌡️ Slightly increase room temperature for comfort")
    
    # Work schedule recommendations
    if user_data['work_schedule'] == 1:
        recommendations.append("🌙 Use blackout curtains for daytime sleep")
        recommendations.append("😴 Try to maintain consistent sleep schedule on days off")
    
    # Default recommendations
    recommendations.extend([
        "🌅 Get natural sunlight exposure in the morning",
        "🍽️ Avoid large meals 3 hours before bedtime",
        "📖 Create a consistent bedtime routine",
        "💤 Keep bedroom quiet and dark"
    ])
    
    return recommendations[:8]  # Return top 8 recommendations

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">😴 AI Sleep Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown("### Get personalized sleep recommendations based on your lifestyle and health data")
    
    # Load model
    model_package = load_model()
    
    if model_package is None:
        st.stop()
    
    # Display model info
    with st.expander("ℹ️ Model Information"):
        info = model_package['model_info']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Samples", f"{info['training_samples']:,}")
        with col2:
            st.metric("Features Used", info['features'])
        with col3:
            st.metric("Model Accuracy", f"{info['quality_r2']:.1%}")
    
    # Sidebar for user inputs
    st.sidebar.header("👤 Enter Your Information")
    
    with st.sidebar.form("user_data_form"):
        st.subheader("Personal Details")
        age = st.number_input("Age", min_value=18, max_value=80, value=30)
        gender = st.selectbox("Gender", ["Female", "Male"])
        weight = st.number_input("Weight (kg)", min_value=40, max_value=120, value=70)
        height = st.number_input("Height (cm)", min_value=150, max_value=200, value=170)
        
        st.subheader("Lifestyle Factors")
        activity_level = st.selectbox(
            "Activity Level", 
            ["Sedentary (little/no exercise)", 
             "Lightly Active (light exercise 1-3 days/week)",
             "Moderately Active (moderate exercise 3-5 days/week)", 
             "Very Active (hard exercise 6-7 days/week)"]
        )
        
        stress_level = st.slider("Stress Level", 1, 10, 5, help="1 = Very relaxed, 10 = Extremely stressed")
        caffeine_intake = st.number_input("Daily Caffeine (mg)", min_value=0, max_value=800, value=150, 
                                        help="1 cup coffee ≈ 95mg, 1 cup tea ≈ 47mg")
        screen_time = st.number_input("Daily Screen Time (hours)", min_value=1, max_value=16, value=8)
        
        st.subheader("Sleep Environment")
        work_schedule = st.selectbox("Work Schedule", ["Regular (9-5)", "Shift Work/Irregular"])
        room_temperature = st.slider("Bedroom Temperature (°C)", 16, 28, 22)
        
        submitted = st.form_submit_button("🔮 Get Sleep Recommendations", use_container_width=True)
    
    if submitted:
        # Prepare user data
        user_data = {
            'age': age,
            'gender_encoded': 1 if gender == "Male" else 0,
            'weight': weight,
            'height': height,
            'bmi': weight / ((height/100) ** 2),
            'activity_level': ["Sedentary (little/no exercise)", 
                             "Lightly Active (light exercise 1-3 days/week)",
                             "Moderately Active (moderate exercise 3-5 days/week)", 
                             "Very Active (hard exercise 6-7 days/week)"].index(activity_level),
            'stress_level': stress_level,
            'caffeine_intake': caffeine_intake,
            'screen_time': screen_time,
            'work_schedule': 1 if work_schedule == "Shift Work/Irregular" else 0,
            'room_temperature': room_temperature
        }
        
        # Make predictions
        predictions = predict_sleep_recommendations(user_data, model_package)
        
        if predictions:
            # Main results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🎯 Sleep Quality Score</h3>
                    <h1>{predictions['sleep_quality']:.1f}/10</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>⏰ Recommended Sleep</h3>
                    <h1>{predictions['sleep_duration']:.1f} hours</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                bedtime_hour = int(predictions['bedtime_hour'])
                bedtime_min = int((predictions['bedtime_hour'] % 1) * 60)
                bedtime_str = f"{bedtime_hour:02d}:{bedtime_min:02d}"
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🌙 Optimal Bedtime</h3>
                    <h1>{bedtime_str}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Detailed analysis
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📊 Your Sleep Profile Analysis")
                
                # Health factors visualization
                factors = ['Sleep Quality', 'Activity Level', 'Stress Management', 'Screen Time Impact']
                scores = [
                    predictions['sleep_quality'],
                    (user_data['activity_level'] + 1) * 2.5,
                    10 - user_data['stress_level'],
                    max(0, 10 - user_data['screen_time'])
                ]
                
                fig = go.Figure(data=go.Bar(
                    x=factors,
                    y=scores,
                    marker_color=['#FF6B6B' if s < 5 else '#FFE66D' if s < 7 else '#4ECDC4' for s in scores],
                    text=[f"{s:.1f}" for s in scores],
                    textposition='outside'
                ))
                fig.update_layout(
                    title="Sleep Health Factors (0-10 scale)",
                    yaxis_range=[0, 10],
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # BMI and lifestyle metrics
                bmi_category = (
                    "Underweight" if user_data['bmi'] < 18.5 else
                    "Normal" if user_data['bmi'] < 25 else
                    "Overweight" if user_data['bmi'] < 30 else
                    "Obese"
                )
                
                st.metric("BMI", f"{user_data['bmi']:.1f}", delta=bmi_category)
                
            with col2:
                st.subheader("💡 Personalized Recommendations")
                
                recommendations = generate_recommendations(user_data, predictions)
                
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"""
                    <div class="recommendation-box">
                        <strong>{i}.</strong> {rec}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Feature importance
            st.subheader("🎯 Factors Affecting Your Sleep Quality")
            
            importance = model_package['feature_importance']
            importance_df = pd.DataFrame(
                list(importance.items()), 
                columns=['Factor', 'Importance']
            ).sort_values('Importance', ascending=True)
            
            # Map technical names to user-friendly names
            factor_names = {
                'stress_level': 'Stress Level',
                'caffeine_intake': 'Caffeine Intake',
                'screen_time': 'Screen Time',
                'activity_level': 'Activity Level',
                'bmi': 'BMI',
                'age': 'Age',
                'room_temperature': 'Room Temperature',
                'work_schedule': 'Work Schedule',
                'gender_encoded': 'Gender',
                'weight': 'Weight',
                'height': 'Height'
            }
            
            importance_df['Factor'] = importance_df['Factor'].map(factor_names)
            
            fig2 = px.bar(
                importance_df, 
                x='Importance', 
                y='Factor',
                orientation='h',
                title="Model Feature Importance",
                color='Importance',
                color_continuous_scale='viridis'
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
            
            # Sleep schedule visualization
            st.subheader("📅 Recommended Sleep Schedule")
            
            # Create a simple schedule visualization
            bedtime = predictions['bedtime_hour']
            wake_time = (bedtime + predictions['sleep_duration']) % 24
            
            schedule_data = pd.DataFrame({
                'Time': ['Bedtime', 'Wake Time'],
                'Hour': [bedtime, wake_time],
                'Type': ['Sleep', 'Wake']
            })
            
            fig3 = px.bar(
                schedule_data, 
                x='Time', 
                y='Hour',
                color='Type',
                title=f"Your Optimal Sleep Schedule ({predictions['sleep_duration']:.1f} hours)",
                color_discrete_map={'Sleep': '#4ECDC4', 'Wake': '#FFE66D'}
            )
            fig3.update_layout(yaxis=dict(range=[0, 24], title="Hour (24-hour format)"))
            st.plotly_chart(fig3, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Disclaimer:** This AI system provides general sleep recommendations based on lifestyle factors. 
    For serious sleep disorders or persistent sleep issues, please consult with a healthcare professional or sleep specialist.
    
    **About the Model:** Trained on synthetic data simulating real sleep patterns and validated for accuracy.
    """)

if __name__ == "__main__":
    main()