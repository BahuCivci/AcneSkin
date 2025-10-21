from flask import Flask, request, session
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this to a random secret key

# In-memory user storage for demonstration purposes
users = {}

def register_user(username, password):
    if username in users:
        return False  # User already exists
    users[username] = generate_password_hash(password)
    return True

def authenticate_user(username, password):
    if username not in users:
        return False
    return check_password_hash(users[username], password)

def is_logged_in():
    return 'username' in session

def login(username):
    session['username'] = username

def logout():
    session.pop('username', None)