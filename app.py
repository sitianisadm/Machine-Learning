
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Memuat pipeline Random Forest yang sudah disimpan
model = joblib.load("model_personality.pkl")

# Nama fitur harus sesuai dengan data saat training
feature_columns = [
    "Age",
    "Gender",
    "Education",
    "Introversion Score",
    "Sensing Score",
    "Thinking Score",
    "Judging Score",
    "Interest"
]

@app.route("/")
def home():
    return jsonify({
        "message": "Personality Prediction API aktif"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Mengambil data JSON dari request
        data = request.get_json()

        if not isinstance(data, dict):
            return jsonify({
                "error": "Kirim data dalam format JSON object."
            }), 400

        # Memeriksa apakah semua fitur tersedia
        missing = [
            col for col in feature_columns
            if col not in data
        ]

        if missing:
            return jsonify({
                "error": "Fitur belum lengkap",
                "missing_features": missing
            }), 400

        # Mengubah data menjadi DataFrame
        input_data = pd.DataFrame(
            [[data[col] for col in feature_columns]],
            columns=feature_columns
        )

        # Pipeline menjalankan preprocessing dan prediksi
        prediction = model.predict(input_data)

        return jsonify({
            "prediction": str(prediction[0])
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)
