import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta
import os
from supabase import create_client
import time
import pytz
import traceback
import sys

print("Starting import of modules: SUCCESS")

class WeatherPredictor:
    def __init__(self, supabase_url, supabase_key, api_key, location, models_dir='models'):
        """
        Initialize WeatherPredictor with real-time data integration
        
        Args:
            supabase_url (str): Supabase project URL
            supabase_key (str): Supabase API key
            api_key (str): Visual Crossing API key
            location (str): Location for weather prediction
            models_dir (str): Directory containing trained models
        """
        print(f"[DEBUG] Initializing WeatherPredictor with location: {location} and models_dir: {models_dir}")
        print(f"[DEBUG] Current working directory: {os.getcwd()}")
        print(f"[DEBUG] Directory contents: {os.listdir('.')}")
        
        if os.path.exists(models_dir):
            print(f"[DEBUG] Models directory exists: {models_dir}")
            print(f"[DEBUG] Models directory contents: {os.listdir(models_dir)}")
        else:
            print(f"[ERROR] Models directory does not exist: {models_dir}")
        
        self.models_dir = models_dir
        print("[DEBUG] Initializing Supabase client")
        self.supabase = create_client(supabase_url, supabase_key)
        print("[DEBUG] Supabase client initialized")
        self.api_key = api_key
        self.location = location
        print("[DEBUG] About to load models")
        self.load_models()
        print("[DEBUG] Models loaded successfully")
        
        # Set timezone to IST
        self.timezone = pytz.timezone('Asia/Kolkata')
        print("[DEBUG] WeatherPredictor initialization complete")

    def load_models(self):
        """Load pre-trained models from local directory"""
        print(f"[DEBUG] Loading models from {self.models_dir}")
        try:
            print(f"[DEBUG] Loading temp_model from {os.path.join(self.models_dir, 'temp_model.joblib')}")
            self.temp_model = joblib.load(os.path.join(self.models_dir, 'temp_model.joblib'))
            print("[DEBUG] temp_model loaded")
            
            print(f"[DEBUG] Loading weather_model from {os.path.join(self.models_dir, 'weather_model.joblib')}")
            self.weather_model = joblib.load(os.path.join(self.models_dir, 'weather_model.joblib'))
            print("[DEBUG] weather_model loaded")
            
            print(f"[DEBUG] Loading conditions_model from {os.path.join(self.models_dir, 'conditions_model.joblib')}")
            self.conditions_model = joblib.load(os.path.join(self.models_dir, 'conditions_model.joblib'))
            print("[DEBUG] conditions_model loaded")
            
            print(f"[DEBUG] Loading scaler from {os.path.join(self.models_dir, 'scaler.joblib')}")
            self.scaler = joblib.load(os.path.join(self.models_dir, 'scaler.joblib'))
            print("[DEBUG] scaler loaded")
            
            print(f"[DEBUG] Loading label_encoder from {os.path.join(self.models_dir, 'label_encoder.joblib')}")
            self.label_encoder = joblib.load(os.path.join(self.models_dir, 'label_encoder.joblib'))
            print("[DEBUG] label_encoder loaded")
            
            print("[DEBUG] All models loaded successfully")
        except FileNotFoundError as e:
            print(f"[ERROR] Model loading error: {e}")
            print(f"[DEBUG] Stack trace: {traceback.format_exc()}")
            raise
        except Exception as e:
            print(f"[ERROR] Unexpected error loading models: {e}")
            print(f"[DEBUG] Stack trace: {traceback.format_exc()}")
            raise

    def fetch_current_sensor_data(self):
        """Fetch latest sensor data from Supabase real-time database"""
        print("[DEBUG] Fetching current sensor data from Supabase")
        try:
            print("[DEBUG] Executing query to fetch sensor data")
            response = self.supabase.table('sensor_data')\
                .select('*')\
                .order('created_at', desc=True)\
                .limit(1)\
                .execute()
            
            print(f"[DEBUG] Query executed, response status: {response.status_code if hasattr(response, 'status_code') else 'unknown'}")
            
            if response.data:
                data = response.data[0]
                print(f"[DEBUG] Sensor data retrieved: {data}")
                return {
                    'temperature': float(data['temperature']),
                    'humidity': float(data['humidity']),
                    'pressure': float(data['pressure']),
                    'uv_index': float(data['uv_index'])
                }
            print("[ERROR] No sensor data available")
            raise Exception("No sensor data available")
        except Exception as e:
            print(f"[ERROR] Error fetching sensor data: {e}")
            print(f"[DEBUG] Stack trace: {traceback.format_exc()}")
            raise

    def fetch_api_data(self):
        print(f"[DEBUG] Fetching API data for location: {self.location}")
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{self.location}"
        params = {
            'unitGroup': 'metric',
            'key': self.api_key,
            'contentType': 'json'
        }
        
        try:
            print(f"[DEBUG] Making API request to {url}")
            response = requests.get(url, params=params)
            print(f"[DEBUG] API response status code: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            print("[DEBUG] API response successfully parsed as JSON")
            
            current = data.get('currentConditions', {})
            print(f"[DEBUG] Current conditions from API: {current}")
            
            result = {
                'windspeed': float(current.get('windspeed', 0) or 0),
                'winddir': float(current.get('winddir', 0) or 0),
                'cloudcover': float(current.get('cloudcover', 0) or 0),
                'visibility': float(current.get('visibility', 10) or 10),
                'rain_level': float(current.get('precip', 0) or 0),
                'sunrise': current.get('sunrise', '06:00:00'),
                'sunset': current.get('sunset', '18:00:00')
            }
            print(f"[DEBUG] Processed API data: {result}")
            return result
        except requests.RequestException as e:
            print(f"[ERROR] API request failed: {e}")
            print(f"[DEBUG] Stack trace: {traceback.format_exc()}")
            raise
        except ValueError as e:
            print(f"[ERROR] Error converting API data: {e}")
            print(f"[DEBUG] Stack trace: {traceback.format_exc()}")
            raise
        except Exception as e:
            print(f"[ERROR] Error fetching API data: {e}")
            print(f"[DEBUG] Stack trace: {traceback.format_exc()}")
            raise

    def store_prediction_inputs(self, sensor_data, api_data):
        """Store the input data used for predictions"""
        print("[DEBUG] Storing prediction inputs")
        try:
            current_time = datetime.now(self.timezone)
            print(f"[DEBUG] Current time in IST: {current_time}")
            
            input_data = {
                'timestamp': current_time.isoformat(),
                'temperature': sensor_data['temperature'],
                'humidity': sensor_data['humidity'],
                'pressure': sensor_data['pressure'],
                'uv_index': sensor_data['uv_index'],
                'windspeed': api_data['windspeed'],
                'winddir': api_data['winddir'],
                'cloudcover': api_data['cloudcover'],
                'visibility': api_data['visibility'],
                'rain_level': api_data['rain_level'],
                'sunrise': api_data['sunrise'],
                'sunset': api_data['sunset']
            }
            print(f"[DEBUG] Input data prepared: {input_data}")
            
            print("[DEBUG] Executing Supabase insert")
            self.supabase.table('prediction_inputs').insert(input_data).execute()
            print(f"[DEBUG] Successfully stored prediction inputs at {current_time}")
            return True
        except Exception as e:
            print(f"[ERROR] Error storing prediction inputs: {e}")
            print(f"[DEBUG] Stack trace: {traceback.format_exc()}")
            return False

    def calculate_wbt(self, temp, humidity):
        """Calculate Wet Bulb Temperature"""
        print(f"[DEBUG] Calculating WBT with temp: {temp}, humidity: {humidity}")
        try:
            es = 6.112 * np.exp(17.67 * temp / (temp + 243.5))
            wbt = temp * np.arctan(0.151977 * np.sqrt(humidity + 8.313659)) + \
                  np.arctan(temp + humidity) - np.arctan(humidity - 1.676331) + \
                  0.00391838 * (humidity) ** (3 / 2) * np.arctan(0.023101 * humidity) - 4.686035
            result = round(float(wbt), 1)
            print(f"[DEBUG] Calculated WBT: {result}")
            return result
        except Exception as e:
            print(f"[ERROR] Error calculating WBT: {e}")
            print(f"[DEBUG] Stack trace: {traceback.format_exc()}")
            return None

    def predict_hourly(self, sensor_data, api_data):
        """Make hourly predictions for the next 8 days using sliding window approach"""
        print("[DEBUG] Starting hourly predictions for next 8 days")
        predictions = []
        current_time = datetime.now(self.timezone)
        print(f"[DEBUG] Current time: {current_time}")
        daily_temps = {}
        
        try:
            # Initialize with current data
            current_temp = sensor_data['temperature']
            current_humidity = sensor_data['humidity']
            current_pressure = sensor_data['pressure']
            current_uv_index = sensor_data['uv_index']
            current_windspeed = api_data['windspeed']
            current_cloudcover = api_data['cloudcover']
            current_winddir = api_data['winddir']
            current_rain_level = api_data['rain_level']
            
            print(f"[DEBUG] Initial conditions - temp: {current_temp}, humidity: {current_humidity}, pressure: {current_pressure}")
            
            # Predict for next 8 days * 24 hours
            print(f"[DEBUG] Starting prediction loop for 8 days (192 hours)")
            for i in range(8 * 24):
                if i % 24 == 0:
                    print(f"[DEBUG] Predicting day {i // 24 + 1}")
                
                future_time = current_time + timedelta(hours=i)
                date_key = future_time.date().isoformat()
                
                features = {
                    'hour': future_time.hour,
                    'day': future_time.day,
                    'month': future_time.month,
                    'day_of_week': future_time.weekday(),
                    'day_of_year': future_time.timetuple().tm_yday,
                    'temp': current_temp,
                    'humidity': current_humidity,
                    'sealevelpressure': current_pressure,
                    'uvindex': current_uv_index,
                    'windspeed': current_windspeed,
                    'cloudcover': current_cloudcover,
                    'winddir': current_winddir
                }

                feature_order = [
                    'hour', 'day', 'month', 'day_of_week', 'day_of_year',
                    'temp', 'humidity', 'sealevelpressure', 'uvindex',
                    'windspeed', 'cloudcover', 'winddir'
                ]
                
                features_df = pd.DataFrame([{key: features[key] for key in feature_order}])
                print(f"[DEBUG] Hour {i}: Created features dataframe with shape {features_df.shape}")
                
                features_scaled = self.scaler.transform(features_df)
                print(f"[DEBUG] Hour {i}: Scaled features")

                # Get temperature predictions
                print(f"[DEBUG] Hour {i}: Predicting temperature")
                temp_predictions = self.temp_model.predict(features_scaled)[0]
                temp = temp_predictions[0]  # Regular temperature
                temp_max = temp_predictions[1]  # Maximum temperature
                temp_min = temp_predictions[2]  # Minimum temperature
                print(f"[DEBUG] Hour {i}: Temperature predictions - temp: {temp}, max: {temp_max}, min: {temp_min}")

                # Get weather predictions
                print(f"[DEBUG] Hour {i}: Predicting weather")
                weather = self.weather_model.predict(features_scaled)[0]
                wind = weather[0]
                humidity = weather[1]
                uv_index = weather[2]
                pressure = weather[3]
                rain_chance = weather[4]
                print(f"[DEBUG] Hour {i}: Weather predictions - wind: {wind}, humidity: {humidity}, rain_chance: {rain_chance}")
                
                # Get condition prediction
                print(f"[DEBUG] Hour {i}: Predicting condition")
                condition_encoded = self.conditions_model.predict(features_scaled)[0]
                condition = self.label_encoder.inverse_transform([condition_encoded])[0]
                print(f"[DEBUG] Hour {i}: Condition prediction - {condition}")

                # Calculate wet bulb temperature
                wbt = self.calculate_wbt(temp, humidity)
                print(f"[DEBUG] Hour {i}: Calculated WBT - {wbt}")
                
                # Track daily min/max temperatures
                if date_key not in daily_temps:
                    daily_temps[date_key] = {'min_temp': temp_min, 'max_temp': temp_max}
                else:
                    daily_temps[date_key]['min_temp'] = min(daily_temps[date_key]['min_temp'], temp_min)
                    daily_temps[date_key]['max_temp'] = max(daily_temps[date_key]['max_temp'], temp_max)
                
                # Create prediction object
                prediction = {
                    'datetime': future_time.isoformat(),
                    'temp': round(float(temp), 1),
                    'temp_max': round(float(temp_max), 1),
                    'temp_min': round(float(temp_min), 1),
                    'condition': condition,
                    'wind': round(float(wind), 1),
                    'humidity': round(float(humidity), 1),
                    'uvIndex': round(float(uv_index), 1),
                    'pressure': round(float(pressure), 1),
                    'rainChance': round(float(rain_chance * 100), 1),
                    'wbt': wbt,
                    'rain_level': round(float(current_rain_level), 1),
                    'wind_direction': round(float(current_winddir), 1),
                    'sunrise': api_data['sunrise'],
                    'sunset': api_data['sunset'],
                    'min_temp': round(float(daily_temps[date_key]['min_temp']), 1),
                    'max_temp': round(float(daily_temps[date_key]['max_temp']), 1),
                    'prediction_made_at': current_time.isoformat()
                }
                
                predictions.append(prediction)
                
                # Update the current values for the next iteration (sliding window)
                current_temp = temp
                current_humidity = humidity
                current_pressure = pressure
                current_uv_index = uv_index
                current_windspeed = wind
                
                # Conditionally update cloudcover and rain_level based on predictions
                if rain_chance > 0.5:  # If rain is likely
                    current_cloudcover = min(100, current_cloudcover + 10)  # Increase cloud cover
                    current_rain_level = max(0, min(25, current_rain_level + (rain_chance * 2)))  # Adjust rain level
                else:
                    current_cloudcover = max(0, current_cloudcover - 5)  # Decrease cloud cover
                    current_rain_level = max(0, current_rain_level - 0.5)  # Decrease rain level
                
                # Randomly adjust wind direction slightly for more realistic predictions
                current_winddir = (current_winddir + np.random.uniform(-10, 10)) % 360

            print(f"[DEBUG] Completed predictions for all 192 hours")
            print(f"[DEBUG] Total predictions generated: {len(predictions)}")
            return predictions
        except Exception as e:
            print(f"[ERROR] Error in predict_hourly: {e}")
            print(f"[DEBUG] Stack trace: {traceback.format_exc()}")
            raise

    def update_predictions_in_supabase(self, predictions):
        """Update predictions in Supabase using upsert operations"""
        print("[DEBUG] Updating predictions in Supabase using upsert")
        try:
            current_time = datetime.now(self.timezone)
            print(f"[DEBUG] Current time: {current_time}")
            
            # Reduced prediction window (optional)
            # predictions = predictions[:120]  # First 5 days instead of 8
            
            # Process in smaller batches for better performance
            batch_size = 20  # Smaller batch size
            total_batches = (len(predictions) + batch_size - 1) // batch_size
            
            print(f"[DEBUG] Upserting {len(predictions)} predictions in {total_batches} batches of {batch_size}")
            
            start_time = time.time()
            for i in range(0, len(predictions), batch_size):
                batch = predictions[i:i + batch_size]
                batch_start = time.time()
                print(f"[DEBUG] Processing batch {i // batch_size + 1}/{total_batches} ({len(batch)} predictions)")
                
                # For each prediction in the batch, do an individual upsert based on datetime
                for pred in batch:
                    try:
                        # Check if record exists
                        resp = self.supabase.table('weather_predictions') \
                            .select('id') \
                            .eq('datetime', pred['datetime']) \
                            .execute()
                        
                        if resp.data and len(resp.data) > 0:
                            # Update existing record
                            self.supabase.table('weather_predictions') \
                                .update(pred) \
                                .eq('datetime', pred['datetime']) \
                                .execute()
                        else:
                            # Insert new record
                            self.supabase.table('weather_predictions') \
                                .insert(pred) \
                                .execute()
                    except Exception as e:
                        print(f"[ERROR] Failed to upsert prediction for {pred['datetime']}: {e}")
                        continue
                
                batch_end = time.time()
                print(f"[DEBUG] Batch {i // batch_size + 1}/{total_batches} completed in {batch_end - batch_start:.2f} seconds")
            
            end_time = time.time()
            print(f"[DEBUG] Successfully updated predictions in {end_time - start_time:.2f} seconds")
            
            # Clean up old records
            try:
                print("[DEBUG] Cleaning up records older than 10 days")
                ten_days_ago = (current_time - timedelta(days=10)).isoformat()
                self.supabase.table('weather_predictions') \
                    .delete() \
                    .lt('datetime', ten_days_ago) \
                    .execute()
                print("[DEBUG] Old records cleanup complete")
            except Exception as e:
                print(f"[ERROR] Failed to clean up old records: {e}")
            
            return True
        except Exception as e:
            print(f"[ERROR] Error in update_predictions_in_supabase: {e}")
            print(f"[DEBUG] Stack trace: {traceback.format_exc()}")
            return False

    def check_database_size(self):
        """Check the database size to avoid hitting free tier limits"""
        try:
            # Note: This is a simplified approach - Supabase doesn't have a direct API to get DB size
            # This is an estimation based on record counts
            print("[DEBUG] Checking database size (estimated)")
            
            # Count weather predictions
            weather_count = self.supabase.table('weather_predictions') \
                .select('id', count='exact') \
                .execute()
            
            # Count inputs
            inputs_count = self.supabase.table('prediction_inputs') \
                .select('id', count='exact') \
                .execute()
            
            # Rough estimation (adjust based on your data)
            weather_size = len(weather_count.data) * 0.5  # KB per record
            inputs_size = len(inputs_count.data) * 0.2  # KB per record
            
            total_size_kb = weather_size + inputs_size
            total_size_mb = total_size_kb / 1024
            
            print(f"[DEBUG] Estimated database size: {total_size_mb:.2f} MB")
            
            # Alert if approaching limit
            if total_size_mb > 400:  # 80% of 500MB limit
                print("[WARNING] Database approaching size limit (80% of free tier)")
            
            return total_size_mb
        except Exception as e:
            print(f"[ERROR] Failed to check database size: {e}")
            return None

    def run_prediction_cycle(self):
        """Run a complete prediction cycle"""
        print("\n" + "="*50)
        print(f"[DEBUG] Starting prediction cycle at {datetime.now(self.timezone)}")
        try:
            # Check database size first
            db_size = self.check_database_size()
            if db_size and db_size > 450:  # 90% of limit
                print("[WARNING] Database size critical - clean up data before proceeding")
                # Implement emergency cleanup or notification logic here
            
            print("[DEBUG] Fetching sensor data...")
            sensor_data = self.fetch_current_sensor_data()
            print(f"[DEBUG] Sensor data: {sensor_data}")
            
            print("[DEBUG] Fetching API data...")
            api_data = self.fetch_api_data()
            print(f"[DEBUG] API data: {api_data}")
            
            print("[DEBUG] Storing prediction inputs...")
            self.store_prediction_inputs(sensor_data, api_data)
            
            print("[DEBUG] Making predictions with sliding window approach...")
            predictions = self.predict_hourly(sensor_data, api_data)
            print(f"[DEBUG] Generated {len(predictions)} predictions")
            
            print("[DEBUG] Updating predictions in database...")
            success = self.update_predictions_in_supabase(predictions)
            
            if success:
                print("[DEBUG] Prediction cycle completed successfully")
            else:
                print("[ERROR] Failed to update predictions in database")
            
            print("="*50 + "\n")
            return success
        except Exception as e:
            print(f"[ERROR] Error in prediction cycle: {e}")
            print(f"[DEBUG] Stack trace: {traceback.format_exc()}")
            print("="*50 + "\n")
            return False

def main():
    """Main function to run the weather prediction service"""
    # Configuration
    print("[DEBUG] Starting main function")
    print("[DEBUG] Reading environment variables")
    
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    API_KEY = os.getenv('API_KEY')
    LOCATION = os.getenv('LOCATION')
    
    print(f"[DEBUG] Environment variables read - SUPABASE_URL: {'set' if SUPABASE_URL else 'NOT SET'}")
    print(f"[DEBUG] Environment variables read - SUPABASE_KEY: {'set' if SUPABASE_KEY else 'NOT SET'}")
    print(f"[DEBUG] Environment variables read - API_KEY: {'set' if API_KEY else 'NOT SET'}")
    print(f"[DEBUG] Environment variables read - LOCATION: {LOCATION if LOCATION else 'NOT SET'}")
    
    if not all([SUPABASE_URL, SUPABASE_KEY, API_KEY, LOCATION]):
        print("[ERROR] Missing required environment variables")
        sys.exit(1)
    
    try:
        print("[DEBUG] Initializing WeatherPredictor")
        predictor = WeatherPredictor(SUPABASE_URL, SUPABASE_KEY, API_KEY, LOCATION)
        print("[DEBUG] WeatherPredictor initialized successfully")
        
        while True:
            print("[DEBUG] Running prediction cycle")
            success = predictor.run_prediction_cycle()
            
            if not success:
                print("[ERROR] Prediction cycle failed, waiting before retry...")
            
            print(f"[DEBUG] Waiting for 4 hours before next prediction cycle...")
            time.sleep(4 * 60 * 60)
            
    except KeyboardInterrupt:
        print("\n[DEBUG] Service stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Critical error in main loop: {e}")
        print(f"[DEBUG] Stack trace: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    print("[DEBUG] Script starting")
    main()