# set FLASK_APP=api.py
# set FLASK_ENV=development
# python -m flask run

from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle as p
import json
import pandas as pd



logistic_regression_file = 'models/logistic_regression.pickle'
logistic_regression_model = p.load(open(logistic_regression_file, 'rb'))

naive_bayes_file = 'models/naive_bayes.pickle'
naive_bayes_model = p.load(open(naive_bayes_file, 'rb'))

random_forest_file = 'models/random_forest.pickle'
random_forest_model = p.load(open(random_forest_file, 'rb'))

decision_tree_file = 'models/decision_tree.pickle'
decision_tree_model = p.load(open(decision_tree_file, 'rb'))

default_model = logistic_regression_model

def create_input(data_dict):
    # Because order counts !
    return [
        data_dict['Administrative'],
        data_dict['Administrative_Duration'],
        data_dict['Informational'],
        data_dict['Informational_Duration'],
        data_dict['ProductRelated'],
        data_dict['ProductRelated_Duration'],
        data_dict['BounceRates'],
        data_dict['ExitRates'],
        data_dict['PageValues'],
        data_dict['SpecialDay'],
        # data_dict['OperatingSystems'],
        # data_dict['Browser'],
        # data_dict['Region'],
        # data_dict['TrafficType'],
        data_dict['Weekend'],
        data_dict['Month_Aug'],
        data_dict['Month_Dec'],
        data_dict['Month_Feb'],
        data_dict['Month_Jul'],
        data_dict['Month_Jun'],
        data_dict['Month_Mar'],
        data_dict['Month_May'],
        data_dict['Month_Nov'],
        data_dict['Month_Oct'],
        data_dict['Month_Sep'],
        data_dict['VisitorType_New_Visitor'],
        data_dict['VisitorType_Other'],
        data_dict['VisitorType_Returning_Visitor'],  
    ] 

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return 'Hello, POST information to /api/ :)'

@app.route('/api', methods=['POST'])
def default_calc():
    data = request.get_json(force=True)
    print("Data received : ", data, '\n\n\n')
    input_list = [create_input(data)]
    prediction = default_model.predict(input_list)[0]
    response = None
    if prediction == 0:
        response = {"response":False}
    else:
        response = {"response":True}
    return response

@app.route('/api/random_forest', methods=['POST'])
def random_forest():
    data = request.get_json(force=True)
    input_list = [create_input(data)]
    prediction = default_model.predict(input_list)[0]
    response = None
    if prediction == 0:
        response = {"response":False}
    else:
        response = {"response":True}
    return response

@app.route('/api/naive_bayes', methods=['POST'])
def naive_bayes():
    data = request.get_json(force=True)
    input_list = [create_input(data)]
    prediction = default_model.predict(input_list)[0]
    response = None
    if prediction == 0:
        response = {"response":False}
    else:
        response = {"response":True}
    return response

@app.route('/api/logistic_regression', methods=['POST'])
def logistic_regression():
    data = request.get_json(force=True)
    input_list = [create_input(data)]
    prediction = default_model.predict(input_list)[0]
    response = None
    if prediction == 0:
        response = {"response":False}
    else:
        response = {"response":True}
    return response

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port='5000')


