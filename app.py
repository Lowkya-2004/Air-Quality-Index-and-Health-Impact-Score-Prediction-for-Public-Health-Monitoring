from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# -----------------------------
# Load trained models
# -----------------------------
aqi_model = joblib.load("aqi.pkl")
health_model = joblib.load("health.pkl")
health_scaler = joblib.load("health_scaler_model.pkl")

# -----------------------------
# AQI Category Function
# -----------------------------
def aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

# -----------------------------
# Feature Health Guidance
# -----------------------------
feature_guidance = {
    "PM2.5": (35, "May cause asthma and breathing difficulty.", "Wear a mask and avoid outdoor exposure."),
    "PM10": (50, "Can cause throat irritation and coughing.", "Limit outdoor activities."),
    "NO2": (40, "Aggravates respiratory conditions.", "Avoid traffic-heavy areas."),
    "SO2": (20, "Triggers breathing problems.", "Stay indoors and close windows."),
    "O3": (100, "Causes chest pain and airway inflammation.", "Avoid outdoor activity during peak hours."),
    "Temperature": ((15, 35), "Extreme temperatures may cause heat or cold stress.", "Stay hydrated and avoid extremes."),
    "Humidity": ((30, 60), "Can worsen respiratory discomfort.", "Use humidifier or dehumidifier."),
    "WindSpeed": (15, "High wind spreads dust and pollutants.", "Reduce outdoor exposure.")
}

# -----------------------------
# Home Route
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Extract inputs
    pm25 = float(data["pm25"])
    pm10 = float(data["pm10"])
    no2 = float(data["no2"])
    so2 = float(data["so2"])
    o3 = float(data["o3"])
    temperature = float(data["temperature"])
    humidity = float(data["humidity"])
    windspeed = float(data["windspeed"])

    # -----------------------------
    # AQI Prediction (PURE ML)
    # -----------------------------
    aqi_input = np.array([[pm25, pm10, no2, so2, o3, temperature, humidity, windspeed]])
    predicted_aqi = float(aqi_model.predict(aqi_input)[0])
    predicted_aqi = max(0, min(predicted_aqi, 500))

    category = aqi_category(predicted_aqi)

    # -----------------------------
    # Health Impact Prediction
    # -----------------------------
    health_input = np.array([[pm25, pm10, no2, so2, o3, temperature, humidity, windspeed, predicted_aqi]])
    health_scaled = health_scaler.transform(health_input)
    health_score = float(health_model.predict(health_scaled)[0])
    health_score = max(0, min(health_score, 100))

  

    # -----------------------------
    # Causes & Precautions (LINE BY LINE)
    # -----------------------------
    effects = []

    inputs = {
        "PM2.5": pm25,
        "PM10": pm10,
        "NO2": no2,
        "SO2": so2,
        "O3": o3,
        "Temperature": temperature,
        "Humidity": humidity,
        "WindSpeed": windspeed
    }

    for feature, value in inputs.items():
        limit, cause, precaution = feature_guidance[feature]

        if isinstance(limit, tuple):
            if value < limit[0] or value > limit[1]:
                effects.append({
                    "cause": f"{feature}: {cause}",
                    "precaution": f"{feature}: {precaution}"
                })
        else:
            if value > limit:
                effects.append({
                    "cause": f"{feature}: {cause}",
                    "precaution": f"{feature}: {precaution}"
                })

    # -----------------------------
    # Response (IMAGE FORMAT)
    # -----------------------------
    return jsonify({
        "predicted_aqi": round(predicted_aqi, 2),   
        "aqi_category": category,                  
        "health_score": round(health_score, 2),    
        "effects": effects                          
    })

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
