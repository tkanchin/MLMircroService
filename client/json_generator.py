'''
@author: Teja Kanchinadam

The below code is used to make json files for train and test sets
'''

import numpy as np
import pandas as pd
from json_tricks.np import dumps
from sklearn.cross_validation import train_test_split

'''
Below method is responsible for pre-processing. This function needs to be changed for every data-set.
The data-set used for this example was Congressional Voting Records from UCI machine learning repository.
Please find the link to the web-site here https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records
'''

'''
Base class or perhaps the only class which is responsible for pre-processing the data, dividng the data 
into test and train sets and converting them into JSON objects.

Parameters
----------------
test_set_size: Float
	percentage in size of test set. default value is 25% or 0.25
cross_validation_folds: Int
	number of folds in the cross-validation set. default value is 5
'''


class MakeJSON(object):

	def __init__(self, test_set_size=0.25, cross_validation_folds=5):
		self.test_set_size = test_set_size
		self.cross_validation_folds = cross_validation_folds
		self.train_dict = {}
		self.test_dict = {}

	def pre_process(self):
		'''reading the data set using a pandas dataframe'''
		df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data', header=None)

		'''
		The length of a feature vector is 16 and all values are either `y` or `n`.
		Replacing all occurences of `n` with 0.
		Replacing all occurences of `y` with 1.
		Missing values are represented with `?`. Replacing all occurences of `?` with NaN.
		'''
		df = df.replace('n', 0)
		df = df.replace('y', 1)
		df = df.replace('?', -1)

		'''
		The first column represent the labels.The 2nd to 16th column indicates features.
		'''

		labels = df[0].as_matrix()
		features = df[[x for x in range(1,len(df.columns))]].as_matrix()
		return features, labels
	
	'''
	The wrapper function which call all other functions from this class.
	'''
	def run_wrapper(self):
		features, labels = self.pre_process()
		x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = self.test_set_size)
		self.make_train_json(x_train, y_train)
		self.make_test_json(x_test, y_test)

	'''
	The below functon makes the training set into train.json

	Format of JSON
	--------------------
	{
		'features': train_features,
		'labels' : train_labels,
		'folds': cross-calidation folds,
		'parameters': parameters for the classifier, default is None
	}
	'''
	def make_train_json(self, x, y):
		self.train_dict = {'features':x, 'labels':y, 'folds': self.cross_validation_folds,'parameters':None}
		tr_json = dumps(self.train_dict)
		target = open('train.json', 'w')
		target.write(tr_json)
		target.close()

	'''
	The below functon makes the training set into train.json

	Format of JSON
	--------------------
	{
		'features': test_features,
		'labels' : test_labels // can be None
	}
	'''

	def make_test_json(self, x, y):
		self.test_dict = {'features':x, 'labels':y}
		ts_json = dumps(self.test_dict)
		target = open('test.json', 'w')
		target.write(ts_json)
		target.close()

'''
Main method in this module, which initializes the MakeJson object and calls the run_wrapper method.
'''
def main():
	obj = MakeJSON()
	obj.run_wrapper()

if __name__ == "__main__":
	main()

