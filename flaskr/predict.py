from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash, Blueprint
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
import importlib

bp = Blueprint('predict', __name__, url_prefix='/predict')
# -------- Print score of my model --------- #
from flaskr import xgboost
from flaskr.xgboost import model, test, train


# ----------input features here-----------
@bp.route('/home')
def home():
    return render_template('cars.html')


#-------------prediction----------
@bp.route('/predict', methods=['GET', 'POST'])
def predict():
	data = {} 
	if request.form:
	# get the form data
		form_data = request.form
		data['form'] = form_data
		predict_mile = float(form_data['predict_mile'])
		predict_modelyear=float(form_data['predict_year'])
		predict_brake=float(form_data['predict_brake'])
		predict_airbag=float(form_data['predict_airbag'])
		predict_camera=float(form_data['predict_camera'])
		predict_control=float(form_data['predict_control'])
		predict_speaker=float(form_data['predict_speaker'])
		predict_video=float(form_data['predict_video'])
		predict_bluetooth=float(form_data['predict_bluetooth'])
		predict_alarm=float(form_data['predict_alarm'])
		predict_navigation=float(form_data['predict_navigation'])
		predict_digital=float(form_data['predict_digital'])
		predict_keyless=float(form_data['predict_keyless'])
		predict_heated=float(form_data['predict_heated'])
		predict_leater=float(form_data['predict_leather'])
		predict_armrest=float(form_data['predict_armrest'])
		predict_total=float(form_data['predict_total'])
		predict_drivetrain=float(form_data['predict_drivetrain'])
		predict_fueltype=float(form_data['predict_fueltype'])
		predict_transmission=float(form_data['predict_transmission'])
		predict_make=float(form_data['predict_make'])
		predict_model=float(form_data['predict_model'])

		input_data = np.array([predict_mile,
		predict_modelyear,
		predict_brake,
		predict_airbag,
		predict_camera,
		predict_control,
		predict_speaker,
		predict_video,
		predict_bluetooth,
		predict_alarm,
		predict_navigation,
		predict_digital,
		predict_keyless,
		predict_heated,
		predict_leater,
		predict_armrest,
		predict_total,
		predict_drivetrain,
		predict_fueltype,
		predict_transmission,
		predict_make, 
		predict_model])

		# get prediction
		load_model=pickle.load(open("model.pkl", "rb"))
		prediction = load_model.predict(input_data.reshape(1, -1))
		prediction = np.exp(prediction[0])

		data['prediction'] = 'Approximate price is {:.1f}'.format(prediction)
	return '{}'.format(data['prediction'])

# @bp.route('/tryform', methods=['GET', 'POST'])
# def tryform():
# 	username = request.form['predict_mile']
# 	return 'it is {}'.format(username)

#-------------prediction----------
@bp.route('/result', methods=['GET', 'POST'])
def result():
	data=predict()
	return render_template('result.html', data=data)

