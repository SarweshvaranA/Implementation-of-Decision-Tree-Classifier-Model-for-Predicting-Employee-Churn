# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries: pandas, sklearn, matplotlib.
2. Read the CSV file using pandas.read_csv then View dataset structure: head(), info(), and check for nulls.
3. Encode the categorical variable "salary" using LabelEncoder then Select feature variables (X) excluding "left" and "department" then Set the target variable (y) as "left".
4. Use train_test_split to split data into training and testing sets (80% train, 20% test).
5. Initialize a Decision Tree Classifier with criterion="entropy" then Fit the model on training data using fit().
6. Predict outcomes for test set using predict().
7. Use accuracy_score from sklearn.metrics to evaluate prediction accuracy.
8. Plot the decision tree using plot_tree() with feature and class names.

## Program:
## Program & Output:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SARWESHVARAN A
RegisterNumber:  212223230198
*/
```

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
data=pd.read_csv("Employee.csv")
data.head()
```
![image](https://github.com/user-attachments/assets/8531432e-f27b-4e25-840b-60b3c20544b0)

```python
data.info()
```
![image](https://github.com/user-attachments/assets/73dc235a-041b-4bb5-b795-63bba3164eb9)


```python
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/786f4fd9-81a5-4388-8669-3052d3222bd1)

```python
data["left"].value_counts()
```
![image](https://github.com/user-attachments/assets/71464079-56b7-4bab-a5a2-f8d808024a32)


```python
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
![image](https://github.com/user-attachments/assets/86e78b7d-213c-4d60-b4c1-1ee67a8c20fc)

```python
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/ea5eec3b-8ffd-4a86-a16b-c347353d6157)

```python
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
![image](https://github.com/user-attachments/assets/0a97dd54-892c-491a-8593-8c88cc499566)

```python
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```

![image](https://github.com/user-attachments/assets/7ff931bb-90a5-4a89-9f58-6e8029d91a1a)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
