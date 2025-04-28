# pip install flask flask-cors pandas scikit-learn aif360 matplotlib

from flask import Flask, request, jsonify
from flask_cors import CORS
from candidate_ranker import get_ranked_candidates
import mysql.connector
import bcrypt


app = Flask(__name__)
CORS(app)  # Allow requests from frontend

dbconn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="bharath",
    database="major"
)
cursor = dbconn.cursor(dictionary=True)


@app.route('/leaderboard', methods=['GET'])
def ranked_candidates():
    ranked_data = get_ranked_candidates()
    return jsonify(ranked_data)

@app.route('/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    existingUser = cursor.fetchone()

    if existingUser and bcrypt.checkpw(password.encode('utf-8'), existingUser['password'].encode('utf-8')):
        return jsonify({"success": True, "role": existingUser['role'], "id": existingUser['id']}) # "user": {"email": existingUser["email"]}})
    else:
        return jsonify({"success": False, "message": "Invalid credentials"}), 401
 
@app.route('/auth/signup', methods=['POST'])
def register():
    data = request.get_json()
    email = data['email']
    password = data['password']
    role = data['role']

    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())


    try:
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        existingUser = cursor.fetchone()
        if existingUser:
            return jsonify({"success": False, "error": "Email already registered"}), 409

        cursor.execute("INSERT INTO users (email, password, role) VALUES (%s, %s, %s)", (email, hashed.decode('utf-8'), role))
        dbconn.commit()
        return jsonify({"success": True})
    except mysql.connector.Error as err:
        return jsonify({"success": False, "error": str(err)}), 500
   
   
@app.route('/applications', methods=['GET'])
def getApplicatons():
    try:
        userId = localStorage.getItem('userId');
        cursor.execute("SELECT * FROM jobRoles")

        dbconn.commit()
        return jsonify({"success": True})
    except mysql.connector.Error as err:
        return jsonify({"success": False, "error": str(err)}), 500
   
   
@app.route('/jobRoles/${jobId}/apply', methods=['POST'])
def getJobRoles():
    data = request.get_json()
    id = data['userId']

    try:
        cursor.execute("ALTER   FROM users WHERE email = %s", (email,))
        
        dbconn.commit()
        return jsonify({"success": True})
    except mysql.connector.Error as err:
        return jsonify({"success": False, "error": str(err)}), 500
   
    
if __name__ == '__main__':
    app.run(debug=True, port=5000)
