import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
h=pd.read_csv(r"C:\Users\madha\Downloads\house\kc_house_data.csv")
h=pd.DataFrame(h)
h.info
h.describe()
h.drop(['id','date'],inplace=True,axis=1)  # drop id and date because its not required 
h.isnull().sum()
X=h.drop(['price'],axis=1)
y=h['price']
from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X_train, y_train)
y_pred=model.predict(X_test)

print("intercept",model.intercept_) # we finding the interception of the value
from sklearn.metrics import r2_score
cff_d=pd.DataFrame(model.coef_,X.columns,columns=["Coefficent"]) # find the coeffiecent
print(cff_d)
print(plt.scatter(y_test,y_pred))
#this graph shows line shape so our model predicted very well
sns.distplot((y_test,y_pred),bins=50)
print("Linear Regression Test Accuracy :", r2_score(y_test, y_pred))
from sklearn import metrics
print("Mean Absoulte Error(MAE):", metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error :", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
