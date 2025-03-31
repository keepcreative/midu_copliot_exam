import json
import argparse
from api import PredictionAPI

def create_labeled_jsonl(input_path="test.jsonl", output_path="labeled_results.jsonl", model_path="/data/app/exam/models/model_20250331_090000.joblib"):
    """
    Process test data and output in JSONL format with labels
    
    Args:
        input_path: Path to input test data in JSONL format
        output_path: Path for output JSONL with labels
        model_path: Path to the trained model
    """
    # Initialize prediction API
    predictor = PredictionAPI(model_path)
    
    # Process each line and write to output
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            # Parse input line
            item = json.loads(line.strip())
            feature = item['feature']
            id_value = item['id']
            
            # Make prediction
            result, error = predictor.predict(feature, id_value)
            if error:
                print(f"Error predicting for {id_value}: {error}")
                continue
                
            # Add label to item
            item['label'] = result['label']
            
            # Write to output file in JSONL format
            outfile.write(json.dumps(item) + '\n')
            
    print(f"Predictions saved to {output_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate labeled JSONL file')
    parser.add_argument('--input', default='test.jsonl', help='Input JSONL file path')
    parser.add_argument('--output', default='labeled_results.jsonl', help='Output JSONL file path')
    parser.add_argument('--model', default='/data/app/exam/models/model_20250331_090000.joblib', 
                        help='Model path')
    
    args = parser.parse_args()
    create_labeled_jsonl(args.input, args.output, args.model)