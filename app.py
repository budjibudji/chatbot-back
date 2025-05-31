import faiss
import pickle
import time
import requests
from sentence_transformers import SentenceTransformer

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from flask_cors import CORS
from sqlalchemy import text

from config import Config

from datetime import datetime

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    discussions = db.relationship('Discussion', backref='user', lazy=True)

class Discussion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    exchanges = db.relationship('Exchange', backref='discussion', lazy=True)

class Exchange(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    discussion_id = db.Column(db.Integer, db.ForeignKey('discussion.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)   # <--- Add this line!

CORS(app)


# Database Setup
bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# Load the sentence-transformers model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index, embeddings, and metadata once
with open("embeddings/index.pkl", "rb") as f:
    index, embeddings, metadatas = pickle.load(f)

# Models

# Check DB connection and create tables
with app.app_context():
    try:
        db.session.execute(text('SELECT 1'))
        print("✅ Database connection successful")
        db.create_all()
        print("✅ Tables created (if they don't exist)")
    except Exception as e:
        print("❌ Error connecting to database:", e)

# Utility functions
def recherche(query, top_k=5):
    vec = model.encode([query])
    scores, ids = index.search(vec, top_k)
    return [metadatas[i] for i in ids[0]]

def interroger_mistral(query, docs):
    context = "\n\n---\n\n".join([
        f"Titre: {d.get('title','')}\nLieu: {d.get('location','')}\nDescription: {d.get('description','')}\nURL: {d.get('url','')}"
        for d in docs
    ])
    
    prompt = f"""
Tu es un expert en ressources humaines et data science.

Voici plusieurs descriptions d'offres d'emploi sélectionnées comme étant les plus proches de la question posée :
{context}

À partir de ces descriptions uniquement, 

{query}
"""

    start_time = time.time()
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "mistral",
        "prompt": prompt,
        "stream": False
    })
    end_time = time.time()
    print(f"⏱ Temps de réponse : {end_time - start_time:.2f} secondes")

    if response.status_code != 200:
        return None, response.text
    return response.json().get("response", "Pas de réponse."), None

# Auth routes
@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'msg': 'Email already exists'}), 400
    hashed_pw = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    user = User(username=data['username'], email=data['email'], password=hashed_pw)
    db.session.add(user)
    db.session.commit()
    token = create_access_token(identity=str(user.id))  # <--- Fix here: convert id to string
    return jsonify({'msg': 'User created successfully', 'token': token}), 201

@app.route('/signin', methods=['POST'])
def signin():
    data = request.json
    user = User.query.filter_by(email=data['email']).first()
    if not user or not bcrypt.check_password_hash(user.password, data['password']):
        return jsonify({'msg': 'Invalid credentials'}), 401
    token = create_access_token(identity=str(user.id))  # <--- Fix here: convert id to string
    return jsonify({'token': token}), 200

# History routes
@app.route('/history', methods=['GET'])
@jwt_required()
def get_history():
    user_id = get_jwt_identity()
    history = History.query.filter_by(user_id=int(user_id)).all()  # Convert back to int if needed
    return jsonify([{ 'prompt': h.prompt, 'response': h.response } for h in history])

@app.route('/history', methods=['POST'])
@jwt_required()
def add_history():
    user_id = get_jwt_identity()
    data = request.json
    new_entry = History(prompt=data['prompt'], response=data['response'], user_id=int(user_id))  # Convert to int
    db.session.add(new_entry)
    db.session.commit()
    return jsonify({'msg': 'History added'}), 201

@app.route("/chatbot", methods=["GET"])
@jwt_required()
def chatbot():
    user_query = request.args.get("query")
    discussion_id = request.args.get("discussion_id", None)
    if not user_query:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    user_id = int(get_jwt_identity())

    # If no discussion_id, create a new discussion
    if discussion_id is None:
        new_discussion = Discussion(user_id=user_id)
        db.session.add(new_discussion)
        db.session.commit()
        discussion_id = new_discussion.id
    else:
        # Validate discussion_id belongs to user
        discussion = Discussion.query.filter_by(id=discussion_id, user_id=user_id).first()
        if not discussion:
            return jsonify({"error": "Invalid discussion_id"}), 400

    # Create Exchange row with question only (no commit yet)
    exchange = Exchange(discussion_id=discussion_id, question=user_query)
    db.session.add(exchange)
    db.session.commit() 


    # Search and get response
    docs = recherche(user_query)
    # mistral_response, error = interroger_mistral(user_query, docs)
    mistral_response, error = interroger_mistral(user_query, docs)

    if error:
        db.session.rollback()
        return jsonify({"error": "Failed to call Mistral API", "details": error}), 500

    # Set the response and commit once
    exchange.response = mistral_response
    db.session.commit()

    return jsonify({
        "discussion_id": discussion_id,
        "query": user_query,
        "response": mistral_response
    })


@app.route('/me', methods=['GET'])
@jwt_required()
def me():
    user_id = get_jwt_identity()
    user = User.query.filter_by(id=int(user_id)).first()
    if not user:
        return jsonify({"msg": "User not found"}), 404
    return jsonify({"username": user.username})


@app.route('/discussions', methods=['GET'])
@jwt_required()
def get_discussion_list():
    user_id = get_jwt_identity()

    discussions = Discussion.query.filter_by(user_id=user_id).order_by(Discussion.created_at.desc()).all()

    result = [{
        'id': d.id,
        'created_at': d.created_at.isoformat()
    } for d in discussions]

    return jsonify(result)

@app.route('/discussions/<int:discussion_id>/messages', methods=['GET'])
@jwt_required()
def get_discussion_messages(discussion_id):
    user_id = get_jwt_identity()

    # Validate ownership
    discussion = Discussion.query.filter_by(id=discussion_id, user_id=user_id).first()
    if not discussion:
        return jsonify({'error': 'Discussion not found or unauthorized'}), 404

    messages = Exchange.query.filter_by(discussion_id=discussion.id).order_by(Exchange.created_at.asc()).all()

    messages_list = [{
        'id': msg.id,
        'question': msg.question,
        'response': msg.response,
        'created_at': msg.created_at.isoformat()
    } for msg in messages]

    return jsonify(messages_list)

# Run app
if __name__ == "__main__":
    app.run(debug=True, port=5000)
