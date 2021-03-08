import pandas
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn import metrics
from datetime import datetime
from matplotlib import pyplot 
import numpy

df = pandas.read_csv("Crimes_-_2001_to_Present.csv", dtype='object')
df.dropna()
dataset = df[(df['Year'] == "2015") | (df['Year'] == "2018") | (df['Year'] == "2019") | (df['Year'] == "2020")]
print(dataset)
dataset = dataset.sample(frac=1).reset_index(drop=True)
dataset = dataset.dropna()
#print(dataset[(dataset['Primary Type'] == "BATTERY")])
dataset['BATTERY'] = numpy.where(dataset['Primary Type']== 'BATTERY', 1,0 )


dataset['Date'] = pandas.to_datetime(dataset['Date'])
dataset['month'] = dataset['Date'].dt.month
dataset['hour'] = dataset['Date'].dt.hour
dataset['hour_slot'] = numpy.select([
	(dataset['hour'] < 4),
	(dataset['hour'] < 8),
	(dataset['hour'] < 12),
	(dataset['hour'] < 16),
	(dataset['hour'] < 20),
	(dataset['hour'] < 24)]
	, [0,1,2,3,4,5])

dataset['hour_slot_0'] = numpy.where(dataset['hour_slot']== 0, 1,0 )
dataset['hour_slot_1'] = numpy.where(dataset['hour_slot']== 1, 1,0 )
dataset['hour_slot_2'] = numpy.where(dataset['hour_slot']== 2, 1,0 )
dataset['hour_slot_3'] = numpy.where(dataset['hour_slot']== 3, 1,0 )
dataset['hour_slot_4'] = numpy.where(dataset['hour_slot']== 4, 1,0 )
dataset['hour_slot_5'] = numpy.where(dataset['hour_slot']== 5, 1,0 )
print(dataset)

target = dataset.iloc[:,22].values
data = dataset.iloc[:,26:32].values
print(data)

machine = linear_model.LogisticRegression()
machine.fit(data, target)

new_data = [[1,0,0,0,0,0],
[0,1,0,0,0,0],
[0,0,1,0,0,0],
[0,0,0,1,0,0],
[0,0,0,0,1,0],
[0,0,0,0,0,1]]
#new_target = machine.predict(new_data)
new_target = machine.predict_proba(new_data)
print(new_target)


kfold_object = KFold(n_splits=4)
kfold_object.get_n_splits(data)

print(kfold_object)

i=0
for training_index, test_index in kfold_object.split(data):
	print(i)
	i=i+1
	print("training: ", training_index)
	print("test: ", test_index)
	data_training = data[training_index]
	data_test = data[test_index]
	target_training = target[training_index]
	target_test = target[test_index]
	# machine = linear_model.LinearRegression()
	machine = linear_model.LogisticRegression()
	machine.fit(data_training, target_training)
	new_target = machine.predict(data_test)
	# print(metrics.r2_score(target_test, new_target))
	print("Accuracy Score: ", metrics.accuracy_score(target_test, new_target))
	print("Confusion Matrix: \n", metrics.confusion_matrix(target_test, new_target))
		
		


	