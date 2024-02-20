import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pumpkin1=data =pd.read_excel(r"C:\Users\madha\Downloads\Pumpkin-Seeds.xlsx")
pumpkin=pd.DataFrame(pumpkin1)
print(pumpkin)
print(pumpkin.shape)
print(pumpkin.info())
print(pumpkin.columns)
print(pumpkin.dtypes)

print(pumpkin.isna().sum())

#print(sns.pairplot(pumpkin))

#Splitting the Dataset

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
print(X.shape, y.shape)

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

#split data for training and testing 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0)
#run simple tree
model = tree.DecisionTreeClassifier()
model = model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print("Simple tree score",accuracy_score(y_test,y_pred))

plt.figure(figsize=(30,20))
print(tree.plot_tree(model,filled=True))

from sklearn.ensemble import RandomForestClassifier
#use random forest with 187 estimators
model_rf = RandomForestClassifier(n_estimators=19,max_depth=6,random_state=0)
model_rf = model_rf.fit(X_train,y_train)

y_predrf = model_rf.predict(X_test)
print('The Testing Accuracy of the random Forest algorithm is ', accuracy_score(y_test, y_predrf))

