# python 3
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os # Input data files are available in the "./unsw-datasoc-sydney-round" directory.

print(os.listdir("./unsw-datasoc-sydney-round"))

# Read in the data with `read_csv()
#test_data = pd.read_csv("./unsw-datasoc-sydney-round/test.csv")

data = pd.read_csv("./unsw-datasoc-sydney-round/cleaned_test.csv")
print("-----")
print(data.describe(include = ['object']))



# import the seaborn module
import seaborn as sns
# import the matplotlib module
import matplotlib.pyplot as plt
# set the background colour of the plot to white
sns.set(style="whitegrid", color_codes=True)
# setting the plot size for all plots
sns.set(rc={'figure.figsize':(11.7,8.27)})
# create a countplot
sns.countplot('sex',data=data,hue = 'smoker_status')
# Remove the top and down margin
sns.despine(offset=10, trim=True)
# display the 
plt.show()

# plotting the violinplot
sns.violinplot(x="smoker_status",y="high_BP", hue="smoker_status", data=data);
plt.show()

sns.violinplot(x="sex",y="high_BP", hue="sex", data=data);
plt.show()

#import the necessary module
from sklearn import preprocessing
# create the Labelencoder object
le = preprocessing.LabelEncoder()
#convert the categorical columns into numeric
encoded_value = le.fit_transform(["red", "blue", "green"])
print("-----")
# encoded value based on alpha order
print(encoded_value)

print("-----")
print(data.dtypes)

print("-----")
print("smoker_status' : ", data.smoker_status.unique())
# print("married' : ", data.married.unique())
# print("age :", data.age.unique())
#convert the categorical columns into numeric
data.high_BP = le.fit_transform(data.high_BP.astype(str))
data.heart_condition_detected_2017 = le.fit_transform(data.heart_condition_detected_2017.astype(str))
data.married = le.fit_transform(data.married.astype(str))
data.BMI = le.fit_transform(data.BMI.astype(str))
data.smoker_status = le.fit_transform(data.smoker_status.astype(str))
data.sex = le.fit_transform(data.sex.astype(str))
data.age = le.fit_transform(data.age.astype(str))
data.job = le.fit_transform(data.job.astype(str))
data.area = le.fit_transform(data.area.astype(str))
#display the initial records
print(data.head())

# dataToCal = data[data.id, data.sex, data.age, data.smoker_status, data.high_BP, data.heart_condition_detected_2017, data.married, data.average_blood_sugar, data.BMI, data.job, data.area]
dataToCal = data.loc[:, ['id', 'sex', 'age', 'smoker_status', 'high_BP', 'heart_condition_detected_2017', 'married', 'average_blood_sugar', 'BMI', 'job', 'area']]
# data = data.drop(['TreatmentA', 'TreatmentB', 'TreatmentC', 'TreatmentD'], axis = 1, inplace = True)
data = data.drop(columns="TreatmentA")
data = data.drop(columns="TreatmentB")
data = data.drop(columns="TreatmentC")
data = data.drop(columns="TreatmentD")

print("-----")
print(data.head())
data['stroke_in_2018'] = None
for col in data.stroke_in_2018:
    data.stroke_in_2018.values[:] = 0

# for key, value in data.iteritems():
#     if key == "stroke_in_2018":
#     	value = 0

#assigning the Oppurtunity Result column as target
target = data.stroke_in_2018
# print("-----")
# print(data.head())

#import the necessary module
from sklearn.model_selection import train_test_split
#split data set into train and test sets
data_train, data_test, target_train, target_test = train_test_split(dataToCal, target.astype('int'), test_size = .40, random_state = 10)
# data_train, data_test, target_train, target_test = train_test_split(data.to_numpy, target, test_size=0.33, random_state=42)

# import the necessary module
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
#create an object of the type GaussianNB
gnb = GaussianNB()
#train the algorithm on training data and predict using the testing data
pred = gnb.fit(data_train, target_train).predict(data_test)
# print(pred.tolist())
#print the accuracy score of the model
print("Naive-Bayes accuracy : ",accuracy_score(target_test, pred, normalize = True))


# print(data.stroke_in_2018)
# from sklearn.svm import LinearSVC
# #create an object of type LinearSVC
# svc_model = LinearSVC(random_state=0)
# #train the algorithm on training data and predict using the testing data
# pred = svc_model.fit(data_train, target_train).predict(data_test)
# #print the accuracy score of the model
# print("LinearSVC accuracy : ",accuracy_score(target_test, pred, normalize = True))

from sklearn.neighbors import KNeighborsClassifier
#create object of the lassifier
neigh = KNeighborsClassifier(n_neighbors=3)
#Train the algorithm
neigh.fit(data_train, target_train)
# predict the response
pred = neigh.predict(data_test)
# evaluate accuracy
print ("KNeighbors accuracy score : ",accuracy_score(target_test, pred))

# export data
export_csv = data.to_csv (r'./unsw-datasoc-sydney-round/completed_test.csv', index = None, header=True)
