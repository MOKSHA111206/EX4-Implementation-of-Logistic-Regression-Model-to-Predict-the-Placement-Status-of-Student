# EX 4 Implementation of Logistic Regression Model to Predict the Placement Status of Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Collect and clean the dataset (handle missing values, encode categorical variables, and scale features).

2.Split the dataset into training and testing sets (e.g., 70% training, 30% testing).

3.Train a logistic regression model on the training data by fitting it to the features and target (placement status).

4.Use the trained model to predict on the test set and evaluate using accuracy, confusion matrix, and other metrics.

5.Adjust model parameters using cross-validation to optimize performance.


## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: AMMINENI MOKSHASREE
RegisterNumber: 2305001001

import pandas as pd
import numpy as np
d=pd.read_csv("/content/ex45Placement_Data (1).csv")
d.head()
data1=d.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
data1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:, :-1]
x
y=data1.iloc[:,-1]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver='liblinear')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred,x_test
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
classification=classification_report(y_test,y_pred)
print("Acuuracy score:",accuracy)
print("\nConfusion matrix:\n",confusion)
print("\nClassification report:\n",classification)
from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()

```

## Output:
![image](https://github.com/user-attachments/assets/f1655e40-202b-40b9-a67a-80d49fd851df)
![image](https://github.com/user-attachments/assets/c873b459-25c7-4a02-acc4-a91ee21e3805)
![image](https://github.com/user-attachments/assets/905b63f8-792b-4b6d-8818-665e9d7ef001)
![image](https://github.com/user-attachments/assets/bf454b88-8695-4b08-a9be-d3fbd44576b2)
![image](https://github.com/user-attachments/assets/d7d0ac60-bc55-4758-ac3e-cf65e568c2cb)
![image](https://github.com/user-attachments/assets/5fa3a5c5-7807-46f5-9dec-56542b99a503)
![image](https://github.com/user-attachments/assets/49c21862-a90c-458d-a6bb-90af6b7dd17a)
![image](https://github.com/user-attachments/assets/ccfb2f58-205a-4bf3-8d6a-399e04066237)
![image](https://github.com/user-attachments/assets/8703c497-462a-4379-abb5-55a21dbf9ef5)
![image](https://github.com/user-attachments/assets/f049046e-8e15-4d50-94cc-a43c8b26490d)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
