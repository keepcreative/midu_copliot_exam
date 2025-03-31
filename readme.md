# Classification Algorithm Competition Solution

This project implements a machine learning model that predicts labels for given features according to competition requirements.

## Project Structure

- `train.py`: Model training and evaluation
- `api.py`: API implementation for prediction service
- `test.py`: Test data processing and label prediction
- `requirements.txt`: Project dependencies

## Setup Instructions

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model**

   ```bash
   python train.py
   ```

3. **Run the API server**

   ```bash
   python api.py
   ```

   The server will start on port 81 by default.

## API Usage

The API accepts POST requests with feature data and returns predicted labels.

**Example request:**

```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{"feature": [40.0, 51.0, 51.0, 39.5, 38.0, 51.0, 51.0, 39.5, 38.0, 51.0, 53.0, 41.5, 38.0, 49.5, 54.0, 42.5, 38.0, 51.5, 59.0, 44.0, 40.0, 53.5, 59.0, 44.0, 39.5, 53.5, 54.5, 43.5, 39.5, 53.5, 54.5, 43.5, 39.5, 53.5, 56.5, 43.5]}'
```

**Example response:**

```json
{
  "label": 3
}
```

## Processing Test Data

To process the test dataset and generate predictions:

```bash
python test.py
```

This will:
1. Load the test data from `test.jsonl`
2. Use the trained model to predict labels
3. Save the results to `test_with_labels.jsonl`

## Model Details

This implementation uses a Random Forest Classifier from scikit-learn, which provides good performance on classification tasks with numerical features. The model is trained on the provided training data and evaluated using accuracy metrics.

## Error Handling

The API implements comprehensive error handling for various scenarios:
- Missing or malformed input data
- Model loading issues
- Server errors
- Invalid endpoints

## Author

liuweizhuo (liuweizhuo@midu.com)
