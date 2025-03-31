"""
Author: marvo marvo@qq.com
Date: 2025-03-31 10:20:46
LastEditors: marvo marvo@qq.com
LastEditTime: 2025-03-31 10:27:12
FilePath: api.py
Description: 这是默认设置,可以在设置》工具》File Description中进行配置
"""

import numpy as np
import joblib
from flask import Flask, request, jsonify


class PredictionAPI:
    def __init__(self, model_path="model.joblib"):
        """Initialize the prediction API with the trained model"""
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the trained model from file"""
        try:
            self.model = joblib.load(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def predict(self, feature, id_value=None):
        """Make prediction for a given feature"""
        try:
            if not isinstance(feature, list):
                raise ValueError("Feature must be a list")

            # Convert feature to numpy array and reshape if needed
            feature_array = np.array(feature).reshape(1, -1)

            # Make prediction
            prediction = int(self.model.predict(feature_array)[0])

            # Return result
            result = {"label": prediction}
            if id_value:
                result["id"] = id_value

            return result, None
        except Exception as e:
            return None, str(e)


def create_app(model_path="/data/app/exam/models/model_20250331_090000.joblib"):
    """Create and configure the Flask application"""
    app = Flask(__name__)
    prediction_api = PredictionAPI(model_path)

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()

            if not data or 'feature' not in data:
                return jsonify({"error": "Missing 'feature' in request"}), 400

            feature = data.get('feature')
            id_value = data.get('id')

            result, error = prediction_api.predict(feature, id_value)
            if error:
                return jsonify({"error": error}), 400

            return jsonify(result), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "Internal server error"}), 500

    return app


if __name__ == "__main__":
    # Create the Flask app
    model_path = "/data/app/exam/models/model_20250331_090000.joblib"
    app = create_app(model_path)

    # Run the Flask app
    print("Starting API server...")
    app.run(host="0.0.0.0", port=5000, debug=True)