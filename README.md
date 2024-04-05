# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Initialize weights randomly.

2.Compute predicted values.

3.Compute gradient of loss function.

4.Update weights using gradient descent. 

## Program
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        
        #calculate predictions
        predictions = (X).dot(theta).reshape(-1,1)
        
        #calculate errors
        errors=(predictions - y ).reshape(-1,1)
        
        #update theta using gradiant descent
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
                                        
data=pd.read_csv("C:/classes/ML/50_Startups.csv")
data.head()

#assuming the lost column is your target variable 'y' 

X = (data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler = StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X)
print(X1_Scaled)

#learn modwl paramerers

theta=linear_regression(X1_Scaled,Y1_Scaled)

#predict target value for a new data
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```
## Output

![image](https://github.com/karthikeyan-R16/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119421232/8e31db8c-d8c5-4807-9894-aa1e2c9a245d)

![image](https://github.com/karthikeyan-R16/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119421232/8c344db6-3c18-4490-8ac7-225fd1c749c1)

![image](https://github.com/karthikeyan-R16/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119421232/170a12f9-a46a-4d37-9ca0-576b9f02577e)


![image](https://github.com/karthikeyan-R16/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119421232/9900283e-2f15-48ef-898d-725804190fc0)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

  
