import pandas as pd    # pandas library used for working with dataset,(# for analyzing, cleaning,manipulating data)#
import numpy as np     # numpy can be used perform mathematical operations on arrays#
import seaborn as sns  # Seaborn is a library for making statistical graphics in Python.#
import matplotlib.pyplot as plt #Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python#
from sklearn import metrics # let you assess the quality of your predictions#
ht=pd.read_csv(r"C:\Users\madha\Downloads\heart\heart.csv") #preparing dataset#
ht=pd.DataFrame(ht) # converting into dataframe
ht.info()          # we are checking the information of the dataset
#Data processsing
ht.isnull().sum()  # we can verify the missed /null values count

X=ht.drop(['target'],axis=1) # we are dropping the value in y axis
y=ht['target'] 
from sklearn.model_selection import train_test_split # split the data into training and test
from sklearn.metrics import confusion_matrix
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)
from sklearn.linear_model import LogisticRegression 
model=LogisticRegression(max_iter=5000) 
model.fit(X_train, y_train)
y_predict=model.predict(X_test)
sns.heatmap(ht.corr(),annot=True)
confusion_matrix(y_test,y_predict)
ht.describe()
from sklearn.metrics import classification_report
print("The Logistic Score Report heart disease:\n",classification_report(y_test,y_predict))
param_grid = [    # hyper tunning#
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]
    }
]


from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(model, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)
best_clf = clf.fit(X,y)

best_clf.best_estimator_

print ("The Hyper Tune Score of Logistic:",f'Accuracy - : {best_clf.score(X,y):.3f}')
# knn algothrim

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
confusion_matrix(y_test,y_predict)
ht.describe()
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
new_data = pd.DataFrame({ # we are giving new data by giving random value from the columns
    'age': [64],
    'sex': [0],
    'cp': [3],
    'trestbps': [146],
     'chol':[234],
     'fbs':[1],
     'restecg':[1],
     'thalach':[150],
     'exang': [1],
     'oldpeak': [2.4],
     'slope': [2],
      'ca':[0],
      'thal':[2],
  })
predicted_ht = model.predict(new_data)
print("\nPredicted heart disease for new data:", predicted_ht)
param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]
    }
]


from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(model, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)
best_clf = clf.fit(X,y)
best_clf.best_estimator_
print ("The Hyper Tune Score of knn:",f'Accuracy - : {best_clf.score(X,y):.3f}')

###############Decesion Tree###################
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model= DecisionTreeClassifier(criterion = 'gini', random_state = 0, max_depth=6)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
yt_pred = model.predict(X_train)
confusion_matrix=confusion_matrix(y_test,y_pred)
print(confusion_matrix)

#print(model.score(X_test,y_test))

print('The Training Accuracy of the algorithm is ', accuracy_score(y_train, yt_pred))
print('The Testing Accuracy of the algorithm is ', accuracy_score(y_test, y_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

plt.figure(figsize=(20,15))
tree.plot_tree(model,filled=True)

predicted_ht = model.predict(new_data)
print("\nPredicted heart disease for new data:", predicted_ht)
param_grid = [    
    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],
    'max_iter' : [100, 1000,2500, 5000]
    }
]
from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(model, param_grid = param_grid, cv = 3, verbose=True, n_jobs=-1)
best_clf = clf.fit(X,y)
best_clf.best_estimator_
print ("The Hyper Tune Score of decision tree:",f'Accuracy - : {best_clf.score(X,y):.3f}')


 