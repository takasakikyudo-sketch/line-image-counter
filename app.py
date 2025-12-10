from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello from Render!"

@app.route('/callback', methods=['POST'])
def callback():
    print("Webhook received:", request.json)
    return "OK"
