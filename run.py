import pandas
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import metrics
from datetime import datetime
from matplotlib import pyplot 
import numpy

df = pandas.read_csv("Crimes_-_2001_to_Present.csv", dtype='object')
df.dropna()
dataset = df[(df['Year'] == 2018) | (df['Year'] == 2019) | (df['Year'] == 2020) ]
dataset = dataset.sample(frac=0.1).reset_index(drop=True)
dataset = dataset.dropna()
dataset.to_csv('Crimes_Sample.csv')

def data_imputation(dataset, impute_target_name, impute_data):
	impute_target = dataset[impute_target_name]
	sub_dataset = pandas.concat([impute_target, impute_data], axis = 1)
	
	data = sub_dataset.loc[:, sub_dataset.columns != impute_target_name].values
	target = dataset[impute_target_name].values

	kfold_object = KFold(n_splits=4)
	kfold_object.get_n_splits(data)
	i=0
	for training_index, test_index in kfold_object.split(data):
		i=i+1
		data_training = data[training_index]
		data_test = data[test_index]
		target_training = target[training_index]
		target_test = target[test_index]
		machine = linear_model.LogisticRegression()
		machine.fit(data_training,target_training)
		new_target = machine.predict(data_test)
		print("Accuracy Score: ", metrics.accuracy_score(target_test, new_target))
		
		


	