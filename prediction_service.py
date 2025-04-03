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
import random
import signal
from huggingface_hub import hf_hub_download, snapshot_download

print("Starting import of modules: SUCCESS")

class TimeoutError(Exception):
    """Exception raised when execution time exceeds the limit."""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Execution exceeded time limit")

class WeatherPredictor:
    def __init__(self, supabase_url, supabase_key, api_key, location, models_dir='models', hf_repo_id=None):
        """
        Initialize WeatherPredictor with Hugging Face model storage
        
        Args:
            supabase_url (str): Supabase project URL
            supabase_key (str): Supabase API key
            api_key (str): Visual Crossing API key
            location (str): Location for weather prediction
            models_dir (str): Directory to store downloaded models
            hf_repo_id (str): Hugging Face repository ID for models
        """
        print(f"[DEBUG] Initializing WeatherPredictor with location: {location}")
        print(f"[DEBUG] Models will be stored in: {models_dir}")
        
        self.models_dir = models_dir
        self.hf_repo_id = hf_repo_id or os.getenv('HF_REPO_ID', 'abin-varghese/weather_models')
        print(f"[DEBUG] Using Hugging Face repo: {self.hf_repo_id}")
        
        os.makedirs(self.models_dir, exist_ok=True)
        
        print("[DEBUG] Initializing Supabase client")
        self.supabase = create_client(supabase_url, supabase_key)
        print("[DEBUG] Supabase client initialized")
        self.api_key = api_key
        self.location = location
        
        # Set timezone to IST
        self.timezone = pytz.timezone('Asia/Kolkata')
        # Rate limiting parameters for free tier
        self.min_db_delay = 0.5  # minimum delay in seconds
        self.max_db_delay = 2.0  # maximum delay in seconds
        
        # Load models
        print("[DEBUG] About to load models")
        self.load_models()
        print("[DEBUG] Models loaded successfully")
        print("[DEBUG] WeatherPredictor initialization complete")

    def download_model(self, filename):
        """Download a single model from Hugging Face Hub"""
        print(f"[DEBUG] Downloading model: {filename}")
        try:
            model_path = hf_hub_download(
                repo_id=self.hf_repo_id,
                filename=filename,
                cache_dir=self.models_dir,
                local_dir=self.models_dir,
                local_dir_use_symlinks=False
            )
            print(f"[DEBUG] Model downloaded to: {model_path}")
            return model_path
        except Exception as e:
            print(f"[ERROR] Error downloading model {filename}: {e}")
            print(f"[DEBUG] Stack trace: {traceback.format_exc()}")
            raise

    def load_models(self):
        """Load pre-trained models from Hugging Face Hub"""
        print(f"[DEBUG] Loading models from Hugging Face Hub: {self.hf_repo_id}")
        try:
            # List of model files we need
            model_files = [
                'temp_model.joblib',
                'weather_model.joblib',
                'conditions_model.joblib',
                'scaler.joblib',
                'label_encoder.joblib'
            ]
            
            # Download all models (this will cache them locally)
            print("[DEBUG] Starting model download from Hugging Face Hub")
            snapshot_download(
                repo_id=self.hf_repo_id,
                local_dir=self.models_dir,
                allow_patterns="*.joblib",
                local_dir_use_symlinks=False
            )
            print("[DEBUG] Model download completed")
            
            # Load each model
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
        except Exception as e:
            print(f"[ERROR] Unexpected error loading models: {e}")
            print(f"[DEBUG] Stack trace: {traceback.format_exc()}")
            raise

    def fetch_current_sensor_data(self):
        """Fetch latest sensor data from Supabase real-time database"""
        print("[DEBUG] Fetching current sensor data from Supabase")
        try:
            print("[DEBUG] Executing query to fetch sensor data")
            # Add delay to avoid rate limiting
            self._apply_rate_limiting_delay()
            
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
            raise Exception("No sensor data available in the database")
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

    def _apply_rate_limiting_delay(self):
        """Apply a random delay to avoid hitting rate limits"""
        delay = random.uniform(self.min_db_delay, self.max_db_delay)
        print(f"[DEBUG] Rate limiting: Waiting {delay:.2f} seconds before database operation")
        time.sleep(delay)

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
            
            print("[DEBUG] Executing Supabase insert with rate limiting")
            self._apply_rate_limiting_delay()
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
                    # Removed 'prediction_date' since this column doesn't exist in the database
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

    def group_predictions_by_date(self, predictions):
        """Group predictions by date for batch processing"""
        print("[DEBUG] Grouping predictions by date")
        date_groups = {}
        for pred in predictions:
            # Extract date from datetime field
            pred_date = pred['datetime'].split('T')[0]
            if pred_date not in date_groups:
                date_groups[pred_date] = []
            date_groups[pred_date].append(pred)
        
        return date_groups

    def delete_predictions_for_date(self, date):
        """Delete all predictions for a specific date using datetime string comparison"""
        print(f"[DEBUG] Deleting predictions for date: {date}")
        try:
            self._apply_rate_limiting_delay()
            # Use datetime field for filtering instead of prediction_date
            # Convert date to datetime range
            start_datetime = f"{date}T00:00:00"
            end_datetime = f"{date}T23:59:59"
            
            # Delete predictions between start and end datetime
            self.supabase.table('weather_predictions').delete().gte('datetime', start_datetime).lte('datetime', end_datetime).execute()
            print(f"[DEBUG] Successfully deleted predictions for date: {date}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to delete predictions for {date}: {e}")
            print(f"[DEBUG] Stack trace: {traceback.format_exc()}")
            # Continue even if delete fails
            return False

    def insert_predictions_in_batches(self, predictions_by_date):
        """Insert predictions in batches to avoid hitting rate limits"""
        print("[DEBUG] Inserting new predictions with rate limiting")
        
        try:
            for date, predictions in predictions_by_date.items():
                print(f"[DEBUG] Processing day {date} with {len(predictions)} predictions")
                
                # Delete existing predictions for this date
                self.delete_predictions_for_date(date)
                
                # Calculate batch size based on number of predictions
                # Smaller batches for free tier
                batch_size = min(6, max(1, len(predictions) // 4))
                num_batches = (len(predictions) + batch_size - 1) // batch_size
                
                print(f"[DEBUG] Processing day {date} with {len(predictions)} predictions in {num_batches} batches")
                
                # Process in batches
                for i in range(0, len(predictions), batch_size):
                    batch_num = (i // batch_size) + 1
                    print(f"[DEBUG] Processing batch {batch_num}/{num_batches} for day {date}")
                    batch = predictions[i:i + batch_size]
                    
                    # Rate limiting
                    self._apply_rate_limiting_delay()
                    
                    # Insert batch
                    try:
                        self.supabase.table('weather_predictions').insert(batch).execute()
                        print(f"[DEBUG] Successfully inserted batch {batch_num}/{num_batches} for {date}")
                    except Exception as e:
                        print(f"[ERROR] Failed to insert batch {batch_num}/{num_batches} for {date}: {e}")
                        # Continue with next batch even if this one fails
                        continue
                
                # Additional delay between days to reduce load
                time.sleep(2)
            
            return True
        except Exception as e:
            print(f"[ERROR] Error in batch insert: {e}")
            print(f"[DEBUG] Stack trace: {traceback.format_exc()}")
            return False

    def update_predictions_in_supabase(self, predictions):
        """Update predictions in Supabase using batch processing"""
        print("[DEBUG] Updating predictions in Supabase with improved error handling and rate limiting")
        try:
            current_time = datetime.now(self.timezone)
            print(f"[DEBUG] Current time: {current_time}")
            
            # Group predictions by date for more efficient processing
            predictions_by_date = self.group_predictions_by_date(predictions)
            print(f"[DEBUG] Grouped predictions into {len(predictions_by_date)} days")
            
            # Insert predictions in batches
            success = self.insert_predictions_in_batches(predictions_by_date)
            
            if success:
                print(f"[DEBUG] Successfully updated all predictions at {current_time}")
            else:
                print(f"[WARNING] Some predictions may not have been updated correctly")
            
            return success
        except Exception as e:
            print(f"[ERROR] Error updating predictions: {e}")
            print(f"[DEBUG] Stack trace: {traceback.format_exc()}")
            return False

    def run_prediction_cycle(self):
        """Run a complete prediction cycle"""
        print("\n" + "="*50)
        cycle_start_time = datetime.now(self.timezone)
        print(f"[DEBUG] Starting prediction cycle at {cycle_start_time}")
        try:
            print("[DEBUG] Fetching sensor data...")
            sensor_data = self.fetch_current_sensor_data()
            print(f"[DEBUG] Sensor data: {sensor_data}")
            
            print("[DEBUG] Fetching API data...")
            api_data = self.fetch_api_data()
            print(f"[DEBUG] API data: {api_data}")
            
            print("[DEBUG] Storing prediction inputs...")
            self.store_prediction_inputs(sensor_data, api_data)
            
            print("[DEBUG] Making predictions with sliding window approach...")
            prediction_start = datetime.now()
            predictions = self.predict_hourly(sensor_data, api_data)
            prediction_end = datetime.now()
            prediction_duration = (prediction_end - prediction_start).total_seconds()
            print(f"[DEBUG] Generated {len(predictions)} predictions in {prediction_duration:.2f} seconds")
            
            print("[DEBUG] Updating predictions in database...")
            db_update_start = datetime.now()
            success = self.update_predictions_in_supabase(predictions)
            db_update_end = datetime.now()
            db_update_duration = (db_update_end - db_update_start).total_seconds()
            
            cycle_end_time = datetime.now(self.timezone)
            total_duration = (cycle_end_time - cycle_start_time).total_seconds()
            
            if success:
                print(f"[DEBUG] Prediction cycle completed successfully in {total_duration:.2f} seconds")
                print(f"[DEBUG] Database update took {db_update_duration:.2f} seconds ({(db_update_duration/total_duration)*100:.1f}% of total time)")
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
    """Main function to run the weather prediction service - now with execution timeout"""
    # Configuration
    print("[DEBUG] Starting main function")
    
    # Set up timeout (30 minutes = 1800 seconds)
    timeout_duration = 30 * 60
    print(f"[DEBUG] Setting up execution timeout of {timeout_duration} seconds (30 minutes)")
    
    # Set up the alarm signal handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_duration)
    
    print("[DEBUG] Reading environment variables")
    
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    API_KEY = os.getenv('API_KEY')
    LOCATION = os.getenv('LOCATION')
    HF_TOKEN = os.getenv('HF_TOKEN')
    HF_REPO_ID = os.getenv('HF_REPO_ID', 'abin-varghese/weather_models')
    
    print(f"[DEBUG] Environment variables read - SUPABASE_URL: {'set' if SUPABASE_URL else 'NOT SET'}")
    print(f"[DEBUG] Environment variables read - SUPABASE_KEY: {'set' if SUPABASE_KEY else 'NOT SET'}")
    print(f"[DEBUG] Environment variables read - API_KEY: {'set' if API_KEY else 'NOT SET'}")
    print(f"[DEBUG] Environment variables read - LOCATION: {LOCATION if LOCATION else 'NOT SET'}")
    print(f"[DEBUG] Environment variables read - HF_TOKEN: {'set' if HF_TOKEN else 'NOT SET'}")
    print(f"[DEBUG] Environment variables read - HF_REPO_ID: {HF_REPO_ID}")
    
    if not all([SUPABASE_URL, SUPABASE_KEY, API_KEY, LOCATION]):
        print("[ERROR] Missing required environment variables")
        sys.exit(1)
    
    try:
        print("[DEBUG] Initializing WeatherPredictor")
        predictor = WeatherPredictor(SUPABASE_URL, SUPABASE_KEY, API_KEY, LOCATION, hf_repo_id=HF_REPO_ID)
        print("[DEBUG] WeatherPredictor initialized successfully")
        
        # Run a single prediction cycle and exit
        print("[DEBUG] Running prediction cycle")
        start_time = datetime.now()
        success = predictor.run_prediction_cycle()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if success:
            print(f"[DEBUG] Prediction cycle completed successfully in {duration:.2f} seconds")
            print("[DEBUG] Prediction service completed successfully. Exiting now.")
            # Exit with success code
            sys.exit(0)
        else:
            print("[ERROR] Prediction cycle failed")
            # Exit with error code
            sys.exit(1)
            
    except TimeoutError:
        print(f"\n[ERROR] Execution timed out after {timeout_duration} seconds (30 minutes)")
        sys.exit(2)
    except KeyboardInterrupt:
        print("\n[DEBUG] Service stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Critical error in main loop: {e}")
        print(f"[DEBUG] Stack trace: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        # Cancel the alarm
        signal.alarm(0)

if __name__ == "__main__":
    print("[DEBUG] Script starting")
    main()
