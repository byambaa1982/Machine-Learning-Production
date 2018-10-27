from flask import Flask, request
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash, Blueprint
import pickle

app = Flask(__name__)


bp = Blueprint('xgboost', __name__, url_prefix='/xgboost')

# ------------ Machine Learning on a Used Car market --------- #
# --------------------GradientBoostingRegression---------------- #
@bp.route('/model')
def model():
	large = pd.read_csv('flaskr/data/used_cars.csv')
	large=large.dropna()
	large=large.drop('index', axis=1)

	X=large.drop('price', axis=1)
	y=large[['price']]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	gbr = GradientBoostingRegressor(n_estimators=90, learning_rate=0.1,
     max_depth=5, random_state=0, loss='ls')

	gbr.fit(X_train, np.log(y_train))
	gbr.score(X_train, np.log(y_train))
	gbr.score(X_test, np.log(y_test))
	y_pred=gbr.predict(X_test)
	y_pred=np.exp(y_pred)

# ------------Pickle my model to use later------------#
	return pickle.dump(gbr, open("model.pkl", "wb"))