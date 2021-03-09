import pandas
from sklearn import linear_model
from datetime import datetime
import numpy

dataset = pandas.read_csv("Crimes_-_2001_to_Present.csv")   # load the dataset


dataset.dropna(inplace=True)	                 # Drop observations with missing data
print(dataset)


print(dataset.groupby('Primary Type').count())



dataset['BATTERY'] = numpy.where(dataset['Primary Type']=='BATTERY',1,0)
dataset['ASSAULT'] = numpy.where(dataset['Primary Type']=='ASSAULT',1,0)
dataset['CRIMINAL DAMAGE'] = numpy.where(dataset['Primary Type']=='CRIMINAL DAMAGE',1,0)
dataset['THEFT'] = numpy.where(dataset['Primary Type']=='THEFT',1,0)
dataset['NARCOTICS'] =numpy.where( dataset['Primary Type']=='NARCOTICS',1,0)
dataset['GAMBLING'] =numpy.where( dataset['Primary Type']=='GAMBLING',1,0)
dataset['MOTOR VEHICLE THEFT'] =numpy.where( dataset['Primary Type']=='MOTOR VEHICLE THEFT',1,0)

target = dataset['Primary Type'].values     # Get the column 'price' 

data = dataset.iloc[:,7:22,23].values    # Get the columns 'Bronx','Brooklyn' ......


machine = linear_model.LogisticRegression()   # Construct the machine
machine.fit(data, target)   # Fit the data and the target

new_data = [
	[1,0,0,0,0],
	[0,1,0,0,0],
	[0,0,1,0,0],
	[0,0,0,1,0],
	[0,0,0,0,1]
]

new_target = machine.predict(new_data)  
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
		
