from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello from Flask on Vercel!"

def handler(environ, start_response):
    return app.wsgi_app(environ, start_response)
