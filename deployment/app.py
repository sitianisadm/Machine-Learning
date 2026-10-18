
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import csv
import json
from pathlib import Path
from datetime import datetime

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_personality.pkl"
LOG_PATH = BASE_DIR / "monitoring.csv"
model = joblib.load(MODEL_PATH)

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

log_columns = [
    "waktu",
    "prediction",
    "probabilitas",
    "data_input",
    "status"
]


def catat_monitoring(waktu, prediction, probabilities, data, status):
    file_baru = not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=log_columns)

        if file_baru:
            writer.writeheader()

        writer.writerow({
            "waktu": waktu,
            "prediction": prediction,
            "probabilitas": json.dumps(probabilities),
            "data_input": json.dumps(data, ensure_ascii=False),
            "status": status
        })


def validasi_data(data):
    if not isinstance(data, dict):
        return "Setiap data harus berupa JSON object."

    missing = [col for col in feature_columns if col not in data]

    if missing:
        return f"Fitur belum lengkap: {', '.join(missing)}"

    if any(data[col] is None for col in feature_columns):
        return "Data memiliki nilai kosong."

    return None


def proses_prediksi(data, waktu):
    input_data = pd.DataFrame(
        [[data[col] for col in feature_columns]],
        columns=feature_columns
    )

    prediction = str(model.predict(input_data)[0])
    probabilities = {}

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]

        probabilities = {
            str(label): round(float(prob), 4)
            for label, prob in zip(model.classes_, probs)
        }

    catat_monitoring(
        waktu, prediction, probabilities, data, "berhasil"
    )

    return {
        "prediction": prediction,
        "probabilities": probabilities,
        "timestamp": waktu
    }


@app.route("/")
def home():
    return jsonify({
        "message": "Personality Prediction API aktif"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)
        error = validasi_data(data)

        if error:
            return jsonify({"error": error}), 400

        waktu = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

        hasil = proses_prediksi(data, waktu)
        return jsonify(hasil)

    except Exception:
        app.logger.exception("Prediksi gagal")
        return jsonify({"error": "Prediksi gagal diproses."}), 500


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    body = request.get_json(silent=True)

    if not isinstance(body, dict):
        return jsonify({
            "error": "Kirim JSON object dengan key 'data'."
        }), 400

    data_list = body.get("data")

    if not isinstance(data_list, list) or not data_list:
        return jsonify({
            "error": "Key 'data' harus berisi daftar data."
        }), 400

    if len(data_list) > 500:
        return jsonify({
            "error": "Maksimal 500 baris per request."
        }), 400

    errors = []

    for i, data in enumerate(data_list):
        error = validasi_data(data)
        if error:
            errors.append({
                "baris": i + 1,
                "error": error
            })

    if errors:
        return jsonify({
            "error": "Validasi gagal. Tidak ada data yang diprediksi.",
            "details": errors
        }), 400

    try:
        waktu = datetime.now().astimezone().isoformat(
            timespec="seconds"
        )

        # Seluruh data diprediksi menggunakan pipeline yang sama.
        X = pd.DataFrame(
            [[row[col] for col in feature_columns]
             for row in data_list],
            columns=feature_columns
        )

        predictions = model.predict(X)
        probability_rows = (
            model.predict_proba(X)
            if hasattr(model, "predict_proba")
            else [None] * len(data_list)
        )

        results = []

        for data, prediction, probs in zip(
            data_list, predictions, probability_rows
        ):
            probabilities = {}

            if probs is not None:
                probabilities = {
                    str(label): round(float(prob), 4)
                    for label, prob in zip(model.classes_, probs)
                }

            hasil = {
                "prediction": str(prediction),
                "probabilities": probabilities,
                "timestamp": waktu
            }

            catat_monitoring(
                waktu, str(prediction), probabilities,
                data, "berhasil"
            )

            results.append(hasil)

        return jsonify({
            "total_data": len(results),
            "results": results
        })

    except Exception:
        app.logger.exception("Prediksi massal gagal")
        return jsonify({
            "error": "Prediksi massal gagal diproses."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
