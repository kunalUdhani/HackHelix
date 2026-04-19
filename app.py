from flask import Flask, request, jsonify
from predict import AnomalyPredictor
import pandas as pd

app = Flask(__name__)
predictor = AnomalyPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        body = request.get_json()
        if not body:
            return jsonify({"error": "No data provided"}), 400

        # Handle both { "data": [...] } and direct list/dict
        data = body.get("data", body)
        
        predictions = predictor.predict(data)

        # Map predictions to labels
        result = ["ANOMALY" if p == 1 else "NORMAL" for p in predictions]

        return jsonify({
            "status": "success",
            "result": result,
            "predictions_raw": predictions
        })

    except Exception as e:
        return jsonify({
            "status": "failed",
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": predictor.model is not None
    })

if __name__ == "__main__":
    print("Starting Flask API...")
    app.run(host='0.0.0.0', port=5000, debug=True)
