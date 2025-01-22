from flask import Flask, request, jsonify
from predictor import NodePredictor

app = Flask(__name__)
predictor = NodePredictor()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        csv_path = data.get('csv_path')
        if not csv_path:
            return jsonify({'error': 'Missing csv_path parameter'}), 400
            
        prediction = predictor.predict(csv_path)
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 