from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import sys
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    news_text = data.get('text', '')
    api_key = data.get('api_key', '')
    
    # Call your detection function
    try:
        # This is a simplified example - you'll need to adapt it to your actual code
        result = detect_fake_news(news_text, api_key)
        return jsonify({
            'verdict': 'REAL' if 'REAL' in result else 'FAKE',
            'ml_probability': 0.85,  # Extract from your result
            'tfidf_similarity': 0.75,  # Extract from your result
            'semantic_similarity': 0.82,  # Extract from your result
            'best_match': {
                'title': 'Sample matching article',
                'url': 'https://example.com/article'
            },
            'details': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download-dataset', methods=['POST'])
def download_dataset():
    try:
        # Run your download script
        result = subprocess.run([sys.executable, 'download.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return jsonify({'success': True, 'message': 'Dataset downloaded successfully'})
        else:
            return jsonify({'error': result.stderr}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train-model', methods=['POST'])
def train_model():
    data = request.json
    fake_csv = data.get('fake_csv', 'Fake.csv')
    true_csv = data.get('true_csv', 'True.csv')
    
    try:
        # Call your training function
        model, threshold = train_model(fake_csv, true_csv)
        
        # This is a simplified example - you'll need to return actual metrics
        return jsonify({
            'accuracy': 0.95,
            'precision': 0.94,
            'recall': 0.96,
            'f1_score': 0.95,
            'optimal_threshold': threshold,
            'confusion_matrix': [[950, 50], [40, 960]]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Import your functions from the main script
    from main import detect_fake_news, train_model
    
    app.run(debug=True, port=5000)