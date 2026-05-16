from flask import Flask, render_template, request
import pickle
import numpy as np
import requests

app = Flask(__name__)

# ==================================
# Load trained ML model
# ==================================
with open("model/crop_model.pkl", "rb") as f:
    model = pickle.load(f)

# ==================================
# Crop details
# ==================================
crop_info = {
    "rice": {"profit": "₹50,000 per acre", "water": "High", "duration": "120 days"},
    "wheat": {"profit": "₹40,000 per acre", "water": "Medium", "duration": "110 days"},
    "maize": {"profit": "₹35,000 per acre", "water": "Medium", "duration": "100 days"},
    "cotton": {"profit": "₹60,000 per acre", "water": "Medium", "duration": "150 days"},
    "sugarcane": {"profit": "₹80,000 per acre", "water": "High", "duration": "12 months"},
    "banana": {"profit": "₹70,000 per acre", "water": "High", "duration": "9 months"},
    "mango": {"profit": "₹90,000 per acre", "water": "Medium", "duration": "2 years"},
    "apple": {"profit": "₹1,00,000 per acre", "water": "Medium", "duration": "3 years"},
    "orange": {"profit": "₹95,000 per acre", "water": "Medium", "duration": "2 years"},
    "papaya": {"profit": "₹65,000 per acre", "water": "Medium", "duration": "8 months"},
    "coconut": {"profit": "₹1,50,000 per acre", "water": "High", "duration": "4 years"},
    "pomegranate": {"profit": "₹85,000 per acre", "water": "Low", "duration": "7 months"},
    "watermelon": {"profit": "₹60,000 per acre", "water": "Medium", "duration": "90 days"},
    "grapes": {"profit": "₹1,20,000 per acre", "water": "Medium", "duration": "1 year"},
    "chickpea": {"profit": "₹55,000 per acre", "water": "Low", "duration": "100 days"},
    "kidneybeans": {"profit": "₹52,000 per acre", "water": "Medium", "duration": "95 days"},
    "coffee": {"profit": "₹1,10,000 per acre", "water": "Medium", "duration": "3 years"}
}

# ==================================
# Fertilizer recommendation
# ==================================
fertilizer_info = {
    "rice": "Urea + DAP",
    "wheat": "Urea + Potash",
    "maize": "NPK 20-20-20",
    "banana": "Organic compost + Potash",
    "grapes": "Phosphorus-rich fertilizer",
    "orange": "Potassium + micronutrients",
    "apple": "Nitrogen + compost",
    "coffee": "Organic manure + NPK",
    "mango": "Farmyard manure + Potash",
    "watermelon": "Compost + Potassium"
}

# ==================================
# Market price data
# ==================================
market_info = {
    "rice": {"price": "₹22/kg", "demand": "High", "season": "Monsoon"},
    "wheat": {"price": "₹28/kg", "demand": "Medium", "season": "Winter"},
    "banana": {"price": "₹40/kg", "demand": "High", "season": "All season"},
    "apple": {"price": "₹120/kg", "demand": "High", "season": "Winter"},
    "orange": {"price": "₹60/kg", "demand": "Medium", "season": "Winter"},
    "watermelon": {"price": "₹25/kg", "demand": "High", "season": "Summer"},
    "grapes": {"price": "₹80/kg", "demand": "High", "season": "Summer"},
    "coffee": {"price": "₹350/kg", "demand": "High", "season": "All season"}
}

# ==================================
# Weather API
# ==================================
def get_weather(city):
    api_key = "54f718563a3bb9b8b3facbf7e56600d2"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    try:
        response = requests.get(url, timeout=5)
        data = response.json()

        if data.get("cod") != 200:
            return None

        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        rainfall = data.get("rain", {}).get("1h", 50)

        return temperature, humidity, rainfall

    except:
        return None


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        name = request.form.get("name", "").strip()
        city = request.form.get("city", "").strip()

        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        ph = float(request.form["ph"])

        weather = get_weather(city)

        if weather:
            temperature, humidity, rainfall = weather
        else:
            temperature = float(request.form.get("temperature", 25))
            humidity = float(request.form.get("humidity", 70))
            rainfall = float(request.form.get("rainfall", 50))

        values = [N, P, K, temperature, humidity, ph, rainfall]

        if any(v < 0 for v in values):
            return render_template(
                "index.html",
                prediction_text="Please enter only positive values"
            )

        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        probs = model.predict_proba(data)[0]
        classes = model.classes_

        top3_idx = np.argsort(probs)[-3:][::-1]

        top3 = []
        for idx in top3_idx:
            top3.append({
                "crop": classes[idx],
                "confidence": round(probs[idx] * 100, 2)
            })

        prediction = top3[0]["crop"].lower().strip()

        info = crop_info.get(prediction, {
            "profit": "Good market value",
            "water": "Varies",
            "duration": "Depends on conditions"
        })

        fertilizer = fertilizer_info.get(
            prediction,
            "General NPK fertilizer"
        )

        # ✅ NEW MARKET FEATURE
        market = market_info.get(prediction, {
            "price": "₹50/kg",
            "demand": "Medium",
            "season": "All season"
        })

        image_file = f"{prediction}.jpg"

        return render_template(
            "result.html",
            name=name,
            city=city,
            crop=prediction.title(),
            profit=info["profit"],
            water=info["water"],
            duration=info["duration"],
            fertilizer=fertilizer,
            crop_image=image_file,
            top3=top3,
            temperature=temperature,
            humidity=humidity,
            rainfall=rainfall,
            price=market["price"],
            demand=market["demand"],
            season=market["season"]
        )

    except Exception as e:
        print("ERROR:", e)
        return render_template(
            "index.html",
            prediction_text="Invalid input values. Please check all fields."
        )


if __name__ == "__main__":
    app.run(debug=True)