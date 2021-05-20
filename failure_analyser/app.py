import os
import pandas as pd
from flask import Flask, request, jsonify, Response, send_from_directory

from io import StringIO
import base64
import uuid
from infer_1 import *
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)
def root_dir():  # pragma: no cover
    return os.path.abspath(os.path.dirname(__file__))

def get_file(filename):  # pragma: no cover
    try:
        src = os.path.join(root_dir(), filename)
    except IOError as exc:
        return str(exc)

@app.route('/get-predictions', methods=['POST'])
def get_preds():
    content = request.json
    unique_path = uuid.uuid4()
    temp_base64 = (content["temp_img"])
    test_base64 = (content["test_img"])
    p = get_predictions(temp_base64, test_base64, str(unique_path))
    #return jsonify({'predictions':p}); #return dictionary structure
    return jsonify(p)

@app.route('/testing/<path:path>')
def send_images(path):
    return send_from_directory('testing', path)
    
@app.route('/', methods=['GET'])
def hello_world():
    content = get_file('index.html')
    return Response(content, mimetype="text/html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4002))
    app.run(debug=True,host='0.0.0.0',port=port)
