# pip install flask flask-cors pandas scikit-learn aif360 matplotlib

from flask import Flask, jsonify
from flask_cors import CORS
from candidate_ranker import get_ranked_candidates

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

@app.route('/leaderboard', methods=['GET'])
def ranked_candidates():
    ranked_data = get_ranked_candidates()
    return jsonify(ranked_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
