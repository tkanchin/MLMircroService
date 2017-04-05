from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import json


class Algorithm(object):
	def __init__(self):
		self.clf = xgb.XGBClassifier()

	def train(self, features, labels, folds):
		avg = []
		y_true = []
		y_pred = []
		kf = StratifiedKFold(labels, n_folds=folds)
		for train_index, test_index in kf:
			x_train, x_test = features[train_index], features[test_index]
			y_train, y_test = labels[train_index], labels[test_index]
			self.clf.fit(x_train,y_train)
			YP = self.clf.predict(x_test)
			avg.append(accuracy_score(y_test,YP))
			y_true.append(y_test.tolist())
			y_pred.append(YP.tolist())
		y_pred = [item for sublist in y_pred for item in sublist]
		y_true = [item for sublist in y_true for item in sublist]
		clf_metrics = self.classification_details(classification_report(y_true, y_pred))
		return_json = {'Metrics':clf_metrics, 'Accuracy':np.mean(avg)}
		self.clf = self.clf.fit(features, labels)
		return return_json
    
	def test(self, features, labels=None):
		y_pred = self.clf.predict(features)
		if labels is not None:
			acc = accuracy_score(labels, y_pred)
			clf_metrics = self.classification_details(classification_report(labels, y_pred))
			return {'Predictions':y_pred.tolist(), 'Metrics':clf_metrics, 'Accuracy':acc}
		return {'Predictions':y_pred.tolist()}
    
	def classification_details(self, string):
		string = [x for x in string.split() if x]
		last = string[-7:]
		last = last[3:]
		string = string[4:-7]
		listi = []
		dicti ={}
		for i in range(0, len(string), 5):
			dicti['Class'] = str(string[i])
			dicti['Precision'] = float(string[i + 1])
			dicti['Recall'] = float(string[i + 2])
			dicti['F1-Score'] = float(string[i +3])
			dicti['Support'] = float(string[i+4])
			listi.append(dicti)
			dicti = {}
		dicti['Class'] = 'Total'
		dicti['Precision'] = float(last[0])
		dicti['Recall'] = float(last[1])
		dicti['F1-Score'] = float(last[2])
		dicti['Support'] = float(last[3])
		listi.append(dicti)
		return listi

app = Flask(__name__)

algo = Algorithm()

@app.route('/', methods=['POST'])
def train():
	resp = algo.train(np.array(request.json['features']["__ndarray__"]), np.array(request.json['labels']["__ndarray__"]), request.json['folds'])
	return jsonify(resp)

@app.route('/test', methods=['POST'])
def test():
	resp = algo.test(np.array(request.json['features']["__ndarray__"]), np.array(request.json['labels']["__ndarray__"]))
	return jsonify(resp)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=8080, debug=True)

