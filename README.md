# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas as pd and import the required dataset.

2.Calculate the null values in the dataset.

3.Import the LabelEncoder from sklearn.preprocessing

4.Convert the string values to numeric values.

5.Import train_test_split from sklearn.model_selection.

6.Assign the train and test dataset.

7.Import DecisionTreeRegressor from sklearn.tree.

8.Import metrics from sklearn.metrics.

9.Calculate the MeanSquareError.

10.Apply the metrics to the dataset.

11.Predict the output for the required values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: varsha s
RegisterNumber: 212222220055
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x= data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 2)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)

from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse

r2= metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:

## Head
![image](https://github.com/user-attachments/assets/86f37aa9-8270-4a9b-a547-59be079e0445)

## info
![image](https://github.com/user-attachments/assets/81fb3723-feb0-4fd7-a3fa-7bda8ffdec1d)

## data.isnull().sum()
![image](https://github.com/user-attachments/assets/34b5f110-b02a-4f64-82e2-42bea27c0255)

## salary head
![image](https://github.com/user-attachments/assets/a1d2bd62-6965-447e-ba94-d560398a7009)

## MSE value
![image](https://github.com/user-attachments/assets/551ad3f6-22a1-4f92-8090-7a2f2a594d09)

## r2 value
![image](https://github.com/user-attachments/assets/564e5e36-ad0e-4793-9fdf-95a9c10b60d4)

## data prediction
![image](https://github.com/user-attachments/assets/0c15ab5b-9e61-4a61-a301-c620efcfa9ce)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
