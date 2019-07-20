# python 3
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
from word2number import w2n

import os # Input data files are available in the "./unsw-datasoc-sydney-round" directory.

print(os.listdir("./unsw-datasoc-sydney-round"))

# Read in the data with `read_csv()
# test_data = pd.read_csv("./unsw-datasoc-sydney-round/test.csv")
test_data = pd.read_csv("./unsw-datasoc-sydney-round/test.csv")
print("-----")
test_data.columns = test_data.columns.str.replace(' ', '_')

# prints all columns
# print(test_data.columns)
# prints all columns types
print(test_data.dtypes)

print("-----")
sexNage = test_data.sex_and_age.str.split(',', 1)
conditionAge = sexNage.str.get(0).str.isalpha() | (sexNage.str.get(0).str.lower() == "male") | (sexNage.str.get(0).str.lower() == "female")
print(conditionAge.head(6))
test_data['sex'] = np.where(conditionAge, sexNage.str.get(0), sexNage.str.get(1))
test_data['age'] = np.where(conditionAge, sexNage.str.get(1), sexNage.str.get(0))
conditionMale = (test_data.sex.str.lower() == "m") | (test_data.sex.str.lower() == "male")
test_data.sex = np.where(conditionMale, "M", test_data.sex.str.upper())
conditionFemale = (test_data.sex.str == "f") | (test_data.sex.str.lower() == "female")
test_data.sex = np.where(conditionFemale, "F", test_data.sex.str.upper())
test_data.sex = test_data.sex.str.replace(' ', '')
test_data.age = test_data.age.str.replace('-', ' ')
# ValueError: Type of input is not string!
# test_data.age = w2n.word_to_num(test_data.age.str)

# for key, value in test_data.iteritems():
# 	if key == "age":
# 		w2n.word_to_num(value.str.lower())
test_data = test_data.drop(columns="sex_and_age")

print("-----")
jobNarea = test_data.job_status_and_living_area.str.split('?', 1)
conditionArea = (jobNarea.str.get(0).str.lower() == "city") | (jobNarea.str.get(0).str.lower() == "c") | (jobNarea.str.get(0).str.lower() == "remote") | (jobNarea.str.get(0).str.lower() == "r")
print(conditionArea.head(6))
test_data['job'] = np.where(conditionArea, jobNarea.str.get(1), jobNarea.str.get(0))
test_data['area'] = np.where(conditionArea, jobNarea.str.get(0), jobNarea.str.get(1))
test_data = test_data.drop(columns="job_status_and_living_area")

# test_data.BMI = test_data.BMI.str.replace(r'[^\w\s]+', '')
test_data.smoker_status = test_data.smoker_status.str.replace(r'[^\w\s]+', '')
test_data.job = test_data.job.str.replace(r'[^\w\s]+', '')
test_data.area = test_data.area.str.replace(r'[^\w\s]+', '')

print("-----")
test_data.smoker_status = test_data.smoker_status.str.replace(' ', '')
conditionSmoke = (test_data.smoker_status.notnull() != True) | test_data.smoker_status.isna() | (test_data.smoker_status.str.lower() == "nonsmoker")
print(conditionArea.head(6))
test_data.smoker_status = np.where(conditionSmoke, "non-smoker", test_data.smoker_status.str.lower())

print("-----")
conditionMarried = (test_data.married.notnull() != True) | (test_data.married == 0)
test_data.married = np.where(conditionMarried, "0", "1")

print("-----")
conditionHBP = (test_data.high_BP.notnull() != True)
test_data.high_BP = np.where(conditionHBP, "0", test_data.high_BP)

print("-----")
conditionBMI = (test_data.BMI.notnull() != True)
test_data.BMI = np.where(conditionBMI, "0", test_data.BMI)

print("-----")
conditionHCD = (test_data.heart_condition_detected_2017.notnull() != True)
test_data.heart_condition_detected_2017 = np.where(conditionHCD, "0", test_data.heart_condition_detected_2017)

print("-----")
conditionABS = (test_data.average_blood_sugar.notnull() != True)
test_data.average_blood_sugar = np.where(conditionABS, "0", test_data.average_blood_sugar)

print("-----")
conditionSex = (test_data.sex.notnull() != True)
test_data.sec = np.where(conditionSex, "T", test_data.sex.str.upper())

print("-----")
conditionAge = (test_data.age.notnull() != True) & (test_data.job.str.lower() == "parental_leave")
test_data.age = np.where(conditionAge, "0", test_data.age)

print("-----")
conditionJob = (test_data.job.notnull() != True)
test_data.job = np.where(conditionJob, "unemployed", test_data.job.str.lower())

print("-----")
conditionArea = (test_data.area.notnull() != True)
test_data.area = np.where(conditionArea, "remote", test_data.area.str.lower())

# export data
export_csv = test_data.to_csv (r'./unsw-datasoc-sydney-round/cleaned_test.csv', index = None, header=True)

print("-----")
print(test_data.head())
test_data.BMI = test_data.BMI.astype(str)
test_data.smoker_status = test_data.smoker_status.astype(str)
test_data.sex = test_data.sex.astype(str)
test_data.age = test_data.age.astype(str)
test_data.job = test_data.job.astype(str)
test_data.area = test_data.area.astype(str)
print(test_data.dtypes)
print("-----")
print(test_data.shape)