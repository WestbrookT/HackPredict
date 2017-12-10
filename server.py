from flask import Flask, request, render_template
from size import predict as p
import json


app = Flask(__name__)

@app.route('/')
def index():

    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():

    #data = request.get_json()



    text = request.form['d']

    highlights, score = p(text)

    out = {'values': highlights, 'score': score}



    return json.dumps(out)

if __name__ == '__main__':
    app.run()