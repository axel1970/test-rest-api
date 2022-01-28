from flask import Flask, render_template, request
import pickle
import numpy as np
from senti import text_data_cleaning
from sklearn import utils
from joblib import load
from scipy.sparse import data

model = pickle.load(open('sent.pkl', 'rb'))


app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    
    #model = load('model.joblib')
    #preds = model.predict(["hello nice to meet you"])
    return render_template('index.html')

# @app.route('/', methods=['POST'])
# def predict():
#     if request.method == "POST":
#        inp = request.form.get("entered")
#        model = load('model.joblib')
#        pred = model.predict(inp)
#     return render_template('index.html',message="pred")

@app.route('/predict', methods=['POST'])
def home():
    inp = request.form['entered']
    if(inp.isnumeric()):
        return render_template('index.html', message="Please enter appropriate sentence")
    arr = np.array([inp])
    pred = model.predict(arr)
    if pred[0] == 1:
#       return render_template('after.html', data=pred)
        return render_template('index.html', message="Positive😄")
    elif pred[0] == 0:
        return render_template('index.html', message="Negative☹️")
   


if __name__ == '__main__':
    app.run(port=3000, debug=True)
  
    #text_data_cleaning = text_data_cleaning()
    #utils.save_document(text_data_cleaning)
