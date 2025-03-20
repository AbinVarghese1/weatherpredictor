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
    """Update predictions in Supabase with enhanced debugging"""
    print("[DEBUG] Updating predictions in Supabase")
    try:
        current_time = datetime.now(self.timezone)
        print(f"[DEBUG] Current time: {current_time}")
        print(f"[DEBUG] Current time ISO format: {current_time.isoformat()}")
        
        # Debug the first few predictions to check format
        print(f"[DEBUG] Sample prediction data format (first prediction):")
        if predictions and len(predictions) > 0:
            for key, value in predictions[0].items():
                print(f"[DEBUG] Field: {key}, Type: {type(value)}, Value: {value}")
        
        # Delete future predictions only
        print("[DEBUG] Deleting existing future predictions")
        try:
            print(f"[DEBUG] Delete query: table='weather_predictions', filter=gte('datetime', '{current_time.isoformat()}')")
            delete_response = self.supabase.table('weather_predictions')\
                .delete()\
                .gte('datetime', current_time.isoformat())\
                .execute()
            
            print(f"[DEBUG] Delete response type: {type(delete_response)}")
            print(f"[DEBUG] Delete response dir: {dir(delete_response)}")
            
            if hasattr(delete_response, 'data'):
                print(f"[DEBUG] Delete response data: {delete_response.data}")
            
            if hasattr(delete_response, 'error'):
                print(f"[DEBUG] Delete response error: {delete_response.error}")
                
            if hasattr(delete_response, 'status_code'):
                print(f"[DEBUG] Delete response status code: {delete_response.status_code}")
            
            print(f"[DEBUG] Delete response complete: {delete_response}")
        except Exception as delete_error:
            print(f"[ERROR] Exception during delete operation: {delete_error}")
            print(f"[DEBUG] Delete error traceback: {traceback.format_exc()}")
        
        # Insert new predictions in batches
        batch_size = 50  # Reduced batch size for testing
        total_batches = (len(predictions) + batch_size - 1) // batch_size
        print(f"[DEBUG] Inserting {len(predictions)} new predictions in {total_batches} batches of {batch_size}")
        
        successful_batches = 0
        failed_batches = 0
        
        for i in range(0, len(predictions), batch_size):
            batch_num = i // batch_size + 1
            batch = predictions[i:i + batch_size]
            batch_start_index = i
            batch_end_index = min(i + batch_size, len(predictions))
            
            print(f"[DEBUG] Processing batch {batch_num}/{total_batches} (indices {batch_start_index}-{batch_end_index-1}, {len(batch)} predictions)")
            
            try:
                print(f"[DEBUG] Batch {batch_num}: Executing insert query")
                insert_start_time = time.time()
                
                insert_response = self.supabase.table('weather_predictions').insert(batch).execute()
                
                insert_end_time = time.time()
                insert_duration = insert_end_time - insert_start_time
                
                print(f"[DEBUG] Batch {batch_num}: Insert took {insert_duration:.2f} seconds")
                print(f"[DEBUG] Batch {batch_num}: Response type: {type(insert_response)}")
                
                if hasattr(insert_response, 'data'):
                    # Log first few items to avoid overwhelming logs
                    data_sample = insert_response.data[:2] if insert_response.data and len(insert_response.data) > 2 else insert_response.data
                    print(f"[DEBUG] Batch {batch_num}: Response data sample: {data_sample}")
                    print(f"[DEBUG] Batch {batch_num}: Response data length: {len(insert_response.data) if insert_response.data else 0}")
                
                if hasattr(insert_response, 'error'):
                    print(f"[ERROR] Batch {batch_num}: Response error: {insert_response.error}")
                
                if hasattr(insert_response, 'status_code'):
                    print(f"[DEBUG] Batch {batch_num}: Response status code: {insert_response.status_code}")
                
                successful_batches += 1
                print(f"[DEBUG] Batch {batch_num}: Insert completed successfully")
                
            except Exception as batch_error:
                failed_batches += 1
                print(f"[ERROR] Batch {batch_num}: Exception during insert: {batch_error}")
                print(f"[DEBUG] Batch {batch_num}: Insert error traceback: {traceback.format_exc()}")
                
                # Try to get more details about the batch that failed
                print(f"[DEBUG] Batch {batch_num}: Examining problematic batch:")
                try:
                    for j, pred in enumerate(batch):
                        # Log a sample from a problematic prediction if batch is large
                        if j < 3 or j > len(batch) - 3:
                            problem_keys = []
                            for key, value in pred.items():
                                if value is None:
                                    problem_keys.append(key)
                                elif key == 'condition' and not isinstance(value, str):
                                    problem_keys.append(f"{key}:{type(value)}")
                                elif key != 'condition' and not isinstance(value, (int, float, str)):
                                    problem_keys.append(f"{key}:{type(value)}")
                                    
                            if problem_keys:
                                print(f"[DEBUG] Batch {batch_num}, Item {j}: Potential problem with keys: {problem_keys}")
                                print(f"[DEBUG] Batch {batch_num}, Item {j}: Values for problem keys: {[(k, pred.get(k)) for k in problem_keys]}")
                except Exception as analysis_error:
                    print(f"[ERROR] Batch {batch_num}: Error while analyzing problematic batch: {analysis_error}")
        
        print(f"[DEBUG] Batch summary: {successful_batches} successful, {failed_batches} failed out of {total_batches} total batches")
        
        if failed_batches == 0:
            print(f"[DEBUG] Successfully updated all predictions at {current_time}")
            return True
        else:
            print(f"[ERROR] Some batches failed during update: {failed_batches} out of {total_batches}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Unhandled error updating predictions: {e}")
        print(f"[DEBUG] Update error traceback: {traceback.format_exc()}")
        return False

    def run_prediction_cycle(self):
        """Run a complete prediction cycle"""
        print("\n" + "="*50)
        print(f"[DEBUG] Starting prediction cycle at {datetime.now(self.timezone)}")
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