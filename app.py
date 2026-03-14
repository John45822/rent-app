from flask import Flask, request, render_template_string
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("rent_classifier_model.pkl")


# -----------------------------
# Prescriptive Function
# -----------------------------
def recommend_action(prediction):

    if prediction == 1:
        return {
            "risk": "High Rent",
            "recommendation": "Verify tenant financial capability before approval",
            "explanation": "High rental cost may require stronger income stability."
        }

    else:
        return {
            "risk": "Low Rent",
            "recommendation": "Proceed with normal rental approval process",
            "explanation": "Lower rental cost presents lower financial risk."
        }


# -----------------------------
# HOME PAGE
# -----------------------------
@app.route("/")
def home():

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Apartment Rent Classifier</title>
    </head>

    <body>

    <h1>🏠 Apartment Rent Classification System</h1>

    <p>
    This system predicts whether an apartment has <b>Low Rent or High Rent</b>
    based on its listing features.
    </p>

    <a href="/predict">
    <button>Start Classification</button>
    </a>

    </body>
    </html>
    """

    return render_template_string(html)


# -----------------------------
# PREDICTION PAGE
# -----------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():

    prediction = None
    risk = None
    recommendation = None
    explanation = None

    if request.method == "POST":

        bathrooms = float(request.form["bathrooms"])
        bedrooms = float(request.form["bedrooms"])
        square_feet = float(request.form["square_feet"])
        latitude = float(request.form["latitude"])
        longitude = float(request.form["longitude"])

        price_type = request.form["price_type"]
        has_photo = request.form["has_photo"]
        pets_allowed = request.form["pets_allowed"]

        cityname = request.form["cityname"]
        state = request.form["state"]

        input_data = pd.DataFrame({
            "bathrooms":[bathrooms],
            "bedrooms":[bedrooms],
            "square_feet":[square_feet],
            "latitude":[latitude],
            "longitude":[longitude],
            "price_type":[price_type],
            "has_photo":[has_photo],
            "pets_allowed":[pets_allowed],
            "cityname":[cityname],
            "state":[state]
        })

        prediction = model.predict(input_data)[0]

        action = recommend_action(prediction)

        risk = action["risk"]
        recommendation = action["recommendation"]
        explanation = action["explanation"]


    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enter Apartment Features</title>
    </head>

    <body>

    <h1>Enter Apartment Features</h1>

    <form method="POST">

    <label>Bathrooms</label>
    <input type="number" name="bathrooms" min="0" max="5" value="1"><br><br>

    <label>Bedrooms</label>
    <input type="number" name="bedrooms" min="0" max="5" value="1"><br><br>

    <label>Square Feet</label>
    <input type="number" name="square_feet" min="200" max="5000" value="800"><br><br>

    <label>Latitude</label>
    <input type="text" name="latitude" value="40.0"><br><br>

    <label>Longitude</label>
    <input type="text" name="longitude" value="-73.0"><br><br>

    <label>Price Type</label>
    <select name="price_type">
    <option>Monthly</option>
    <option>Weekly</option>
    </select><br><br>

    <label>Has Photo</label>
    <select name="has_photo">
    <option>Yes</option>
    <option>No</option>
    </select><br><br>

    <label>Pets Allowed</label>
    <select name="pets_allowed">
    <option>Yes</option>
    <option>No</option>
    </select><br><br>

    <label>City Name</label>
    <input type="text" name="cityname" value="New York"><br><br>

    <label>State</label>
    <input type="text" name="state" value="NY"><br><br>

    <button type="submit">Predict Rent Category</button>

    </form>

    {% if prediction is not none %}

    <h2>Prediction Result</h2>

    {% if prediction == 1 %}
    <p>💰 High Rent Apartment</p>
    {% else %}
    <p>🏡 Low Rent Apartment</p>
    {% endif %}

    <h3>Risk Level</h3>
    <p>{{ risk }}</p>

    <h3>Recommended Action</h3>
    <p>{{ recommendation }}</p>

    <h3>Explanation</h3>
    <p>{{ explanation }}</p>

    {% endif %}

    <br>
    <a href="/">
    <button>Back to Home</button>
    </a>

    </body>
    </html>
    """

    return render_template_string(
        html,
        prediction=prediction,
        risk=risk,
        recommendation=recommendation,
        explanation=explanation
    )


# -----------------------------
# RUN SERVER (RENDER READY)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)