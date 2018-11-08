from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash, Blueprint
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
import importlib

bp = Blueprint('score', __name__, url_prefix='/score')
# -------- Print score of my model --------- #
from flaskr import xgboost
from flaskr.xgboost import model, test, train

@bp.route('/score', methods=('GET', 'POST'))

def score():
	load_model=pickle.load(open("model.pkl", "rb"))
	train_score=train()
	test_score=test()
	return render_template('blog/score.html',output=train_score, test_score=test_score)
