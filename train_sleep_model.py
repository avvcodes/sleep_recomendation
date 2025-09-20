# train_sleep_model.py
"""
Complete Sleep Recommendation Model Training Script
Run this first to create the model file for Streamlit app
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import warnings
warnings.filterwarnings('ignore')

def create_sleep_dataset(n_samples=5000):
    """Create a realistic sleep dataset"""
    np.random.seed(42)
    
    # Generate base features
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'weight': np.random.normal(70, 15, n_samples).clip(40, 120),
        'height': np.random.normal(170, 10, n_samples).clip(150, 200),
        'activity_level': np.random.randint(0, 4, n_samples),  # 0=Sedentary, 3=Very Active
        'stress_level': np.random.randint(1, 11, n_samples),
        'caffeine_intake': np.random.exponential(150, n_samples).clip(0, 800),
        'screen_time': np.random.normal(8, 3, n_samples).clip(1, 16),
        'work_schedule': np.random.choice([0, 1], n_samples),  # 0=Regular, 1=Shift work
        'room_temperature': np.random.normal(22, 3, n_samples).clip(16, 28),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate BMI
    df['bmi'] = df['weight'] / ((df['height']/100) ** 2)
    
    # Encode gender (0=Female, 1=Male)
    df['gender_encoded'] = (df['gender'] == 'Male').astype(int)
    
    # Create realistic sleep quality based on multiple factors
    sleep_quality = (
        8.5  # Base quality
        - (df['stress_level'] - 5) * 0.4  # Stress impact
        - (df['caffeine_intake'] / 100) * 0.3  # Caffeine impact
        - np.maximum(0, df['screen_time'] - 6) * 0.2  # Excess screen time
        + (df['activity_level']) * 0.3  # Activity benefit
        - np.maximum(0, df['bmi'] - 25) * 0.1  # BMI impact
        - df['work_schedule'] * 0.8  # Shift work penalty
        - np.abs(df['room_temperature'] - 20) * 0.1  # Temperature impact
        + (df['age'] < 30) * 0.3 - (df['age'] > 60) * 0.4  # Age factors
        + np.random.normal(0, 0.5, n_samples)  # Random variation
    )
    
    df['sleep_quality'] = np.clip(sleep_quality, 1, 10)
    
    # Create sleep duration recommendation (6-9 hours)
    sleep_duration = (
        8  # Base duration
        + (df['age'] > 65) * 0.5  # Elderly need slightly more
        - (df['age'] < 25) * 0.3  # Young adults can manage with less
        + (df['stress_level'] > 7) * 0.5  # High stress needs more sleep
        + (df['activity_level'] > 2) * 0.3  # Active people need more recovery
        + np.random.normal(0, 0.3, n_samples)
    )
    
    df['recommended_sleep_duration'] = np.clip(sleep_duration, 6, 9)
    
    # Create bedtime recommendation (based on chronotype and lifestyle)
    bedtime_hour = (
        22  # Base bedtime (10 PM)
        + (df['age'] > 50) * -0.5  # Older people sleep earlier
        + (df['work_schedule']) * 2  # Shift workers later
        + (df['stress_level'] > 7) * 0.5  # Stressed people delay sleep
        + np.random.normal(0, 1, n_samples)
    )
    
    df['recommended_bedtime_hour'] = np.clip(bedtime_hour, 20, 24)  # 8 PM to 12 AM
    
    return df

def train_sleep_model():
    """Train the sleep recommendation model"""
    print("📊 Creating sleep dataset...")
    df = create_sleep_dataset()
    
    # Prepare features
    feature_columns = [
        'age', 'gender_encoded', 'weight', 'height', 'bmi',
        'activity_level', 'stress_level', 'caffeine_intake', 
        'screen_time', 'work_schedule', 'room_temperature'
    ]
    
    print(f"📋 Dataset shape: {df.shape}")
    print(f"🎯 Features: {feature_columns}")
    
    # Prepare multiple targets
    X = df[feature_columns]
    y_quality = df['sleep_quality']
    y_duration = df['recommended_sleep_duration']
    y_bedtime = df['recommended_bedtime_hour']
    
    # Split data
    X_train, X_test, y_quality_train, y_quality_test = train_test_split(
        X, y_quality, test_size=0.2, random_state=42
    )
    _, _, y_duration_train, y_duration_test = train_test_split(
        X, y_duration, test_size=0.2, random_state=42
    )
    _, _, y_bedtime_train, y_bedtime_test = train_test_split(
        X, y_bedtime, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("🤖 Training models...")
    
    # Train models for different predictions
    models = {}
    
    # Sleep Quality Model
    quality_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    quality_model.fit(X_train_scaled, y_quality_train)
    quality_pred = quality_model.predict(X_test_scaled)
    
    # Sleep Duration Model
    duration_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    duration_model.fit(X_train_scaled, y_duration_train)
    duration_pred = duration_model.predict(X_test_scaled)
    
    # Bedtime Model
    bedtime_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    bedtime_model.fit(X_train_scaled, y_bedtime_train)
    bedtime_pred = bedtime_model.predict(X_test_scaled)
    
    # Evaluate models
    print("\n📈 Model Performance:")
    print(f"Sleep Quality - MAE: {mean_absolute_error(y_quality_test, quality_pred):.3f}, R²: {r2_score(y_quality_test, quality_pred):.3f}")
    print(f"Sleep Duration - MAE: {mean_absolute_error(y_duration_test, duration_pred):.3f}, R²: {r2_score(y_duration_test, duration_pred):.3f}")
    print(f"Bedtime - MAE: {mean_absolute_error(y_bedtime_test, bedtime_pred):.3f}, R²: {r2_score(y_bedtime_test, bedtime_pred):.3f}")
    
    # Package everything together
    model_package = {
        'quality_model': quality_model,
        'duration_model': duration_model,
        'bedtime_model': bedtime_model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'feature_importance': dict(zip(feature_columns, quality_model.feature_importances_)),
        'model_info': {
            'training_samples': len(X_train),
            'features': len(feature_columns),
            'quality_r2': r2_score(y_quality_test, quality_pred),
            'duration_r2': r2_score(y_duration_test, duration_pred),
            'bedtime_r2': r2_score(y_bedtime_test, bedtime_pred)
        }
    }
    
    return model_package

def save_model():
    """Train and save the complete model package"""
    print("🚀 Starting Sleep Recommendation Model Training...\n")
    
    # Train model
    model_package = train_sleep_model()
    
    # Save the model
    print("\n💾 Saving model...")
    with open('sleep_model.pkl', 'wb') as f:
        pickle.dump(model_package, f)
    
    print("✅ Model saved as 'sleep_model.pkl'")
    print(f"📊 Model Info: {model_package['model_info']}")
    
    # Test loading
    print("\n🔍 Testing model loading...")
    with open('sleep_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    
    print("✅ Model loaded successfully!")
    print("🎯 Ready for Streamlit deployment!")
    
    return model_package

if __name__ == "__main__":
    model_package = save_model()
    
    print("\n" + "="*50)
    print("NEXT STEPS:")
    print("1. Run: pip install streamlit pandas numpy scikit-learn plotly")
    print("2. Save the Streamlit app code as 'app.py'")
    print("3. Run: streamlit run app.py")
    print("="*50)