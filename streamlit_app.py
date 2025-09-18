import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Simplified version for testing
np.random.seed(42)
tf.random.set_seed(42)

def generate_simple_time_series_data(n_users=20, days_per_user=30):
    """Generate simplified time series sleep data"""
    
    all_data = []
    
    for user_id in range(n_users):
        # User characteristics
        user_chronotype = np.random.choice([0, 1, 2])  # 0=morning, 1=evening, 2=neither
        user_stress_baseline = np.random.uniform(3, 7)
        user_caffeine_sensitivity = np.random.uniform(0.5, 1.5)
        
        for day in range(days_per_user):
            date = datetime.now() - timedelta(days=days_per_user-day)
            
            # Daily variations
            is_weekday = date.weekday() < 5
            
            # Features
            bedtime_hour = 22.5 + np.random.normal(0, 1) + (1 if user_chronotype == 1 else 0)
            sleep_duration = 7.5 + np.random.normal(0, 0.8)
            work_stress = user_stress_baseline + np.random.normal(0, 1) + (1 if is_weekday else -1)
            caffeine_intake = np.random.poisson(2) + (1 if is_weekday else 0)
            exercise_minutes = np.random.exponential(30) if np.random.random() > 0.4 else 0
            screen_time = np.random.exponential(2.5) + (0.5 if not is_weekday else 0)
            
            # Wearable data (simplified)
            heart_rate = 70 + np.random.normal(0, 8)
            heart_rate_var = 45 + np.random.normal(0, 10)
            
            # Environmental
            room_temp = np.random.normal(20, 2)
            noise_level = np.random.exponential(3)
            
            # Calculate sleep quality based on multiple factors
            quality_factors = []
            
            # Sleep duration factor
            quality_factors.append(8 - abs(sleep_duration - 7.5))
            
            # Bedtime factor
            quality_factors.append(8 - abs(bedtime_hour - 22.5) / 2)
            
            # Stress factor
            quality_factors.append(max(0, 8 - work_stress))
            
            # Caffeine factor
            quality_factors.append(max(0, 8 - caffeine_intake * user_caffeine_sensitivity))
            
            # Exercise factor
            if exercise_minutes > 0:
                quality_factors.append(min(8, exercise_minutes / 10))
            else:
                quality_factors.append(4)
            
            # Screen time factor
            quality_factors.append(max(0, 8 - screen_time))
            
            # Environmental factors
            quality_factors.append(8 if 18 <= room_temp <= 22 else max(0, 8 - abs(room_temp - 20)))
            quality_factors.append(max(0, 8 - noise_level))
            
            sleep_quality = np.mean(quality_factors) + np.random.normal(0, 0.5)
            sleep_quality = np.clip(sleep_quality, 0, 10)
            
            all_data.append({
                'user_id': user_id,
                'day': day,
                'date': date,
                'is_weekday': int(is_weekday),
                'bedtime_hour': bedtime_hour,
                'sleep_duration': sleep_duration,
                'work_stress': work_stress,
                'caffeine_intake': caffeine_intake,
                'exercise_minutes': exercise_minutes,
                'screen_time': screen_time,
                'heart_rate': heart_rate,
                'heart_rate_var': heart_rate_var,
                'room_temp': room_temp,
                'noise_level': noise_level,
                'sleep_quality': sleep_quality
            })
    
    return pd.DataFrame(all_data)

class SimpleLSTMPredictor:
    """Simplified LSTM predictor for sleep quality"""
    
    def __init__(self, sequence_length=7):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def prepare_data(self, df):
        """Prepare sequences for LSTM training"""
        
        self.feature_columns = ['bedtime_hour', 'sleep_duration', 'work_stress', 
                               'caffeine_intake', 'exercise_minutes', 'screen_time',
                               'heart_rate', 'heart_rate_var', 'room_temp', 'noise_level']
        
        sequences = []
        targets = []
        user_ids = []
        
        for user_id in df['user_id'].unique():
            user_data = df[df['user_id'] == user_id].sort_values('day')
            
            if len(user_data) > self.sequence_length:
                for i in range(self.sequence_length, len(user_data)):
                    # Get sequence of past days
                    seq = user_data.iloc[i-self.sequence_length:i][self.feature_columns].values
                    sequences.append(seq)
                    
                    # Target is current day's sleep quality
                    targets.append(user_data.iloc[i]['sleep_quality'])
                    user_ids.append(user_id)
        
        return np.array(sequences), np.array(targets), np.array(user_ids)
    
    def build_model(self):
        """Build simplified LSTM model"""
        
        model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(self.sequence_length, len(self.feature_columns))),
            Dropout(0.2),
            LSTM(16),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')  # Output 0-1, will scale to 0-10
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        self.model = model
        return model
    
    def train(self, df, epochs=30):
        """Train the LSTM model"""
        
        print("Preparing sequences...")
        X, y, user_ids = self.prepare_data(df)
        
        print(f"Created {len(X)} sequences")
        print(f"Sequence shape: {X.shape}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1]))
        X_scaled = X_scaled.reshape(X.shape)
        
        # Scale targets to 0-1 range for sigmoid
        y_scaled = y / 10.0
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        
        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Build model
        self.build_model()
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=8,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluate
        test_loss, test_mae = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test MAE: {test_mae * 10:.2f} (on 0-10 scale)")
        
        return history
    
    def predict(self, sequence):
        """Predict sleep quality from a sequence"""
        
        # Scale sequence
        seq_scaled = self.scaler.transform(sequence.reshape(-1, sequence.shape[-1]))
        seq_scaled = seq_scaled.reshape(1, *sequence.shape)
        
        # Predict and scale back to 0-10
        prediction = self.model.predict(seq_scaled, verbose=0)[0][0] * 10
        return prediction

class SimplePersonalizedRecommender:
    """Simplified personalized recommender with feedback learning"""
    
    def __init__(self):
        self.user_feedback = {}  # Store user-specific feedback
        self.recommendation_success_rates = {}
    
    def get_recommendations(self, user_id, current_features, predicted_quality):
        """Get personalized recommendations based on current features"""
        
        recommendations = []
        
        # Initialize user if new
        if user_id not in self.user_feedback:
            self.user_feedback[user_id] = {}
            self.recommendation_success_rates[user_id] = {
                'reduce_caffeine': 0.5,
                'earlier_bedtime': 0.5,
                'more_exercise': 0.5,
                'reduce_screen_time': 0.5,
                'stress_management': 0.5
            }
        
        success_rates = self.recommendation_success_rates[user_id]
        
        # Extract features (assuming same order as feature_columns)
        bedtime, sleep_dur, stress, caffeine, exercise, screen_time = current_features[:6]
        
        # Generate recommendations based on thresholds and success rates
        if caffeine > 2 and success_rates['reduce_caffeine'] > 0.3:
            recommendations.append({
                'type': 'reduce_caffeine',
                'message': f"Consider reducing caffeine from {caffeine:.1f} to 1-2 cups daily",
                'success_rate': success_rates['reduce_caffeine']
            })
        
        if bedtime > 23 and success_rates['earlier_bedtime'] > 0.3:
            recommendations.append({
                'type': 'earlier_bedtime',
                'message': f"Try going to bed 30 minutes earlier (currently {bedtime:.1f})",
                'success_rate': success_rates['earlier_bedtime']
            })
        
        if exercise < 20 and success_rates['more_exercise'] > 0.3:
            recommendations.append({
                'type': 'more_exercise',
                'message': f"Increase daily exercise to 30+ minutes (currently {exercise:.0f})",
                'success_rate': success_rates['more_exercise']
            })
        
        if screen_time > 2 and success_rates['reduce_screen_time'] > 0.3:
            recommendations.append({
                'type': 'reduce_screen_time',
                'message': f"Reduce evening screen time from {screen_time:.1f} to under 2 hours",
                'success_rate': success_rates['reduce_screen_time']
            })
        
        if stress > 6 and success_rates['stress_management'] > 0.3:
            recommendations.append({
                'type': 'stress_management',
                'message': f"Try stress reduction - meditation or yoga (stress: {stress:.1f}/10)",
                'success_rate': success_rates['stress_management']
            })
        
        # Sort by success rate
        recommendations.sort(key=lambda x: x['success_rate'], reverse=True)
        
        return recommendations[:3]  # Top 3
    
    def record_feedback(self, user_id, rec_type, followed, improved):
        """Record user feedback and update success rates"""
        
        if user_id not in self.user_feedback:
            self.user_feedback[user_id] = {}
        
        # Store feedback
        if rec_type not in self.user_feedback[user_id]:
            self.user_feedback[user_id][rec_type] = []
        
        self.user_feedback[user_id][rec_type].append({
            'followed': followed,
            'improved': improved,
            'timestamp': datetime.now()
        })
        
        # Update success rate using exponential moving average
        if followed:
            current_rate = self.recommendation_success_rates[user_id][rec_type]
            new_success = 1.0 if improved else 0.0
            # Weight recent feedback more heavily
            self.recommendation_success_rates[user_id][rec_type] = (
                0.7 * current_rate + 0.3 * new_success
            )
        else:
            # Slight penalty for not following
            self.recommendation_success_rates[user_id][rec_type] *= 0.9
        
        # Keep in reasonable bounds
        self.recommendation_success_rates[user_id][rec_type] = np.clip(
            self.recommendation_success_rates[user_id][rec_type], 0.1, 0.9
        )

def demo_simple_advanced_system():
    """Demo the simplified advanced system"""
    
    print("🚀 SIMPLIFIED ADVANCED SLEEP AI DEMO")
    print("="*50)
    
    # Generate data
    print("📊 Generating time series data...")
    df = generate_simple_time_series_data(n_users=15, days_per_user=20)
    
    print(f"Generated {len(df)} data points for {df['user_id'].nunique()} users")
    print(f"Sleep quality range: {df['sleep_quality'].min():.1f} - {df['sleep_quality'].max():.1f}")
    
    # Train LSTM predictor
    print("\n🧠 Training LSTM predictor...")
    predictor = SimpleLSTMPredictor()
    history = predictor.train(df, epochs=20)
    
    # Initialize recommender
    recommender = SimplePersonalizedRecommender()
    
    # Demo with users
    print("\n🎯 PERSONALIZED PREDICTIONS & RECOMMENDATIONS")
    print("="*50)
    
    test_users = [0, 1, 2]
    
    for user_id in test_users:
        print(f"\n👤 USER {user_id}:")
        print("-" * 20)
        
        # Get user's recent data
        user_data = df[df['user_id'] == user_id].sort_values('day')
        
        if len(user_data) >= 8:  # Need at least 8 days (7 for sequence + 1 current)
            # Get last 7 days as sequence
            recent_data = user_data.tail(8)
            past_sequence = recent_data.iloc[:7][predictor.feature_columns].values
            current_features = recent_data.iloc[-1][predictor.feature_columns].values
            actual_quality = recent_data.iloc[-1]['sleep_quality']
            
            # Predict sleep quality
            predicted_quality = predictor.predict(past_sequence)
            
            print(f"Actual Sleep Quality: {actual_quality:.1f}/10")
            print(f"Predicted Sleep Quality: {predicted_quality:.1f}/10")
            print(f"Prediction Error: {abs(actual_quality - predicted_quality):.1f}")
            
            # Get personalized recommendations
            recommendations = recommender.get_recommendations(
                user_id, current_features, predicted_quality
            )
            
            if recommendations:
                print(f"\nPersonalized Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    success_rate = rec['success_rate'] * 100
                    print(f"  {i}. {rec['message']} (Success rate: {success_rate:.1f}%)")
                
                # Simulate user feedback for learning
                print(f"\n📝 Simulating user feedback...")
                for rec in recommendations:
                    # Higher quality users more likely to follow recommendations
                    follow_prob = 0.6 + (predicted_quality / 20)  # 0.6 to 1.1
                    followed = np.random.random() < follow_prob
                    
                    # If followed, improvement depends on recommendation quality
                    if followed:
                        improve_prob = rec['success_rate']
                        improved = np.random.random() < improve_prob
                    else:
                        improved = False
                    
                    # Record feedback
                    recommender.record_feedback(user_id, rec['type'], followed, improved)
                    
                    status = "✅" if followed and improved else "❌" if not followed else "⚠️"
                    print(f"    {status} {rec['type']}: Followed={followed}, Improved={improved}")
            else:
                print("✅ No recommendations needed - habits look great!")
    
    # Show learning progress
    print(f"\n📈 LEARNING PROGRESS - Success Rates After Feedback:")
    print("="*55)
    
    for user_id in test_users:
        if user_id in recommender.recommendation_success_rates:
            print(f"\n👤 User {user_id} Success Rates:")
            rates = recommender.recommendation_success_rates[user_id]
            for rec_type, rate in rates.items():
                print(f"  • {rec_type.replace('_', ' ').title()}: {rate:.1%}")
    
    return predictor, recommender, df

def test_prediction_accuracy(predictor, df):
    """Test the prediction accuracy of the trained model"""
    
    print(f"\n🎯 TESTING PREDICTION ACCURACY")
    print("="*40)
    
    errors = []
    predictions = []
    actuals = []
    
    # Test on multiple users
    for user_id in df['user_id'].unique()[:5]:  # Test first 5 users
        user_data = df[df['user_id'] == user_id].sort_values('day')
        
        if len(user_data) >= 10:  # Need enough data
            # Test on last few days
            for i in range(7, min(len(user_data), 12)):
                past_seq = user_data.iloc[i-7:i][predictor.feature_columns].values
                actual = user_data.iloc[i]['sleep_quality']
                predicted = predictor.predict(past_seq)
                
                error = abs(actual - predicted)
                errors.append(error)
                predictions.append(predicted)
                actuals.append(actual)
    
    if errors:
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        min_error = np.min(errors)
        
        print(f"Average Prediction Error: {avg_error:.2f}/10")
        print(f"Max Error: {max_error:.2f}/10")
        print(f"Min Error: {min_error:.2f}/10")
        print(f"Predictions within 1.0 point: {sum(1 for e in errors if e <= 1.0) / len(errors):.1%}")
        print(f"Predictions within 2.0 points: {sum(1 for e in errors if e <= 2.0) / len(errors):.1%}")
        
        # Show sample predictions
        print(f"\nSample Predictions:")
        for i in range(min(5, len(actuals))):
            print(f"  Actual: {actuals[i]:.1f}, Predicted: {predictions[i]:.1f}, Error: {abs(actuals[i]-predictions[i]):.1f}")

def create_your_test_case(predictor, recommender):
    """Create a test case with your own data"""
    
    print(f"\n🔬 TEST WITH YOUR OWN DATA")
    print("="*35)
    
    print("Enter your sleep data for the past 7 days:")
    print("(Press Enter for example values)")
    
    try:
        # Collect 7 days of data
        past_week = []
        
        for day in range(1, 8):
            print(f"\n--- Day {day} (most recent = Day 7) ---")
            
            bedtime = input(f"Bedtime hour (e.g., 22.5 for 10:30 PM) [default: 23]: ").strip()
            bedtime = float(bedtime) if bedtime else 23.0
            
            duration = input(f"Sleep duration hours [default: 7.5]: ").strip()
            duration = float(duration) if duration else 7.5
            
            stress = input(f"Work stress level 1-10 [default: 5]: ").strip()
            stress = float(stress) if stress else 5.0
            
            caffeine = input(f"Caffeine cups [default: 2]: ").strip()
            caffeine = float(caffeine) if caffeine else 2.0
            
            exercise = input(f"Exercise minutes [default: 30]: ").strip()
            exercise = float(exercise) if exercise else 30.0
            
            screen = input(f"Evening screen time hours [default: 2]: ").strip()
            screen = float(screen) if screen else 2.0
            
            # Use reasonable defaults for other features
            day_data = [
                bedtime, duration, stress, caffeine, exercise, screen,
                70, 45, 20, 2  # heart_rate, hrv, room_temp, noise_level
            ]
            
            past_week.append(day_data)
        
        # Convert to numpy array
        past_sequence = np.array(past_week)
        
        # Predict tomorrow's sleep
        predicted_quality = predictor.predict(past_sequence)
        
        print(f"\n🔮 PREDICTION FOR TONIGHT:")
        print(f"Predicted Sleep Quality: {predicted_quality:.1f}/10")
        
        if predicted_quality >= 8:
            print("   → Excellent sleep expected! 🌟")
        elif predicted_quality >= 6:
            print("   → Good sleep expected 😊")
        elif predicted_quality >= 4:
            print("   → Fair sleep - room for improvement 🤔")
        else:
            print("   → Poor sleep predicted - needs attention ⚠️")
        
        # Get recommendations based on most recent day
        user_id = 999  # Test user ID
        current_features = past_week[-1]  # Most recent day
        
        recommendations = recommender.get_recommendations(
            user_id, current_features, predicted_quality
        )
        
        if recommendations:
            print(f"\n💡 PERSONALIZED RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                success_rate = rec['success_rate'] * 100
                print(f"  {i}. {rec['message']}")
                print(f"     Success rate for users like you: {success_rate:.1f}%")
        else:
            print(f"\n✅ No specific recommendations - your habits look great!")
        
        return predicted_quality, recommendations
        
    except (ValueError, KeyboardInterrupt):
        print("❌ Invalid input or cancelled. Using example data instead.")
        
        # Use example data
        example_week = np.array([
            [23.5, 7.0, 6, 3, 20, 3, 72, 42, 21, 3],  # Day 1
            [23.0, 7.5, 5, 2, 45, 2, 68, 48, 20, 2],  # Day 2
            [24.0, 6.5, 8, 4, 0, 4, 75, 38, 22, 4],   # Day 3
            [22.5, 8.0, 4, 1, 60, 1, 65, 52, 19, 1],  # Day 4
            [23.5, 7.0, 7, 3, 30, 3, 71, 44, 21, 3],  # Day 5
            [24.5, 6.0, 9, 5, 0, 5, 78, 35, 23, 5],   # Day 6
            [23.0, 7.5, 6, 2, 40, 2, 69, 47, 20, 2],  # Day 7 (today)
        ])
        
        predicted_quality = predictor.predict(example_week)
        print(f"\nUsing example data...")
        print(f"Predicted Sleep Quality: {predicted_quality:.1f}/10")
        
        recommendations = recommender.get_recommendations(999, example_week[-1], predicted_quality)
        if recommendations:
            print(f"\nRecommendations based on example data:")
            for rec in recommendations:
                print(f"• {rec['message']}")

if __name__ == "__main__":
    print("🚀 Starting Simple Advanced Sleep AI Demo...")
    
    # Run the main demo
    predictor, recommender, df = demo_simple_advanced_system()
    
    # Test accuracy
    test_prediction_accuracy(predictor, df)
    
    # Interactive test with your data
    create_your_test_case(predictor, recommender)
    
    print(f"\n✅ DEMO COMPLETED!")
    print("="*40)
    print("🧠 Deep Learning: ✅ LSTM neural network trained")
    print("📈 Time Series: ✅ 7-day sequence analysis") 
    print("👤 Personalization: ✅ Individual success rate tracking")
    print("🔄 Feedback Learning: ✅ Recommendations improve over time")
    print("🎯 Multi-modal: ✅ Lifestyle + physiological + environmental data")
    
    print(f"\n💡 How it works:")
    print("1. LSTM learns temporal patterns from your past 7 days")
    print("2. Recommendations personalized based on what worked for you before")
    print("3. System learns from your feedback to improve future suggestions")
    print("4. Each user gets their own success rate model that adapts over time")