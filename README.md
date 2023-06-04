# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.
```

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: Karthikeyan R
RegisterNumber:  212222240045

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(x,y,theta):
  m=len(y)
  h=x.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
  
data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(x,y,theta)
def gradientDescent(x,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions = x.dot(theta)
    error = np.dot(x.transpose(),(predictions -y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(x,y,theta))

  return theta,J_history
  theta,J_history = gradientDescent(x,y,theta,0.01,1500)
print("h(x) *"+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\theta)$")
plt.title("Cost frunction using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0]for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.ylabel("Profit predictions")

def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))

```
## output :

![ml3](https://github.com/karthikeyan-R16/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119421232/f5480dc3-117c-45df-bae9-063312d7508d)

![ml3 1](https://github.com/karthikeyan-R16/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119421232/dc3d82cd-caca-48db-832d-7756e483b7d1)

![ml3 2](https://github.com/karthikeyan-R16/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119421232/0cfcf9d6-9da2-4f6c-a7d0-441fe5c52d5b)

![ml3 3](https://github.com/karthikeyan-R16/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119421232/b6ce1557-4fbc-4cd3-9acc-0b005c83b0e3)

![ml3 4](https://github.com/karthikeyan-R16/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119421232/15831b3c-c989-4b11-aa6f-a29a7e9d0aa1)

![ml3 5](https://github.com/karthikeyan-R16/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119421232/d6c11f6c-fb62-4d84-b26f-0ad65626b82c)


## Result
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
