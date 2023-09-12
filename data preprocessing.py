#import the packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#import the dataset
dataset = pd.read_csv(r"C:\Users\dell\Documents\DATA SCIENCE,AI & ML\12th\PART-1- CLASS WORK\Data.csv")



#split the dataset into independent and dependent variables as "x" and "y"
x = dataset.iloc[:,:-1].values                                #independent variable
y = dataset.iloc[:,3].values                                  #dependent variable



#impute numerical values of the independent variables(missing value treatment)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
// I THINK IT WILL BE GOOD IF WE MENTION THE STRATERGY WE ARE USING TO TAKE CARE OF MISSING DATA 
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])



#impute categorical values of independent variables(variable transformation)
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
labelencoder_x.fit_transform(x[:,0])
x[:,0] = labelencoder_x.fit_transform(x[:,0])



#impute categorical values of dependent variables(variable transformation)
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
labelencoder_y.fit_transform(y)
y = labelencoder_y.fit_transform(y)



#scale the data(feature scalling)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)



#split the data into training and testing phase(train and test data)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.20, random_state = 0)



#we are done with the data preprocessing 
