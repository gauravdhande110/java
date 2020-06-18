# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 16:58:01 2019

@author: gaura
"""


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import numpy as np

df1 = pd.read_csv("indiadata.csv",engine='python',usecols = ['gname'])
df = pd.read_csv("indiadata.csv",skipinitialspace=True, usecols = ['city','longitude','latitude','attacktype1_txt','targtype1_txt'])

import pandas as pd
df = df[df1.gname != 'Unknown']
df1 = df1[df1.gname != 'Unknown']
from sklearn.preprocessing import LabelEncoder 
  
le = LabelEncoder() 
bf1=df1  
bf1['gname']= le.fit_transform(df1['gname']) 
bf=df
bf['city']= le.fit_transform(df['city']) 
bf['attacktype1_txt']= le.fit_transform(df['attacktype1_txt']) 
bf['targtype1_txt']= le.fit_transform(df['targtype1_txt']) 
#print(bf)
#print(bf1)



from sklearn.model_selection import train_test_split
X_train , X_test ,y_train,y_test = train_test_split(bf,bf1,test_size = 0.3 , random_state = 0 )
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)

# Train the model using the training sets
model.fit(X_train,y_train)

#Predict Output
predictedknn= model.predict(X_test) # 0:Overcast, 2:Mild
from sklearn.metrics import confusion_matrix
conknn = confusion_matrix(y_test,predictedknn)
#print("Accuracyknn:",metrics.accuracy_score(y_test,predictedknn)*100)

from sklearn.tree import DecisionTreeClassifier
declf = DecisionTreeClassifier()
declf = declf.fit(X_train,y_train)
predictedDT= declf.predict(X_test)
conDT = confusion_matrix(y_test,predictedDT)
#print("AccuracyDT:",metrics.accuracy_score(y_test,predictedDT)*100)

from sklearn.naive_bayes import GaussianNB
modelNB = GaussianNB()
modelNB.fit(X_train,y_train)
predictedNB= modelNB.predict(X_test)
conNB = confusion_matrix(y_test,predictedNB)
                         
# Model Accuracy, how often is the classifier correct?
#print("AccuracyNB:",metrics.accuracy_score(y_test,predictedNB)*100)
from sklearn.svm import SVC
clf = SVC(gamma='auto')
clf.fit(X_train, y_train) 
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
predictedSVC = clf.predict(X_test) # 0:Overcast, 2:Mild


from sklearn.ensemble import RandomForestClassifier
forestclass=RandomForestClassifier(n_estimators=50)
#Train the model using the training sets y_pred=clf.predict(X_test)
forestclass.fit(X_train,y_train)
predictedRF=forestclass.predict(X_test)




label=['K-nn acc','K-nn pre','K-nn recall', 'DTree acc','DTree pre','DTree recall',
       'NB acc','NB pre' ,'NB recall', 'SVM acc','SVM pre',
       'SVM recall','RanFor acc','RanFor pre' ,'RanFor recall']

acc= [
    metrics.accuracy_score(y_test,predictedknn)*100,
    precision_score(y_test,predictedknn,average='weighted')*100,
    recall_score(y_test,predictedknn,average='weighted')*100,
    metrics.accuracy_score(y_test,predictedDT)*100,
    precision_score(y_test,predictedDT,average='weighted')*100,
    recall_score(y_test,predictedDT,average='weighted')*100,
    metrics.accuracy_score(y_test,predictedNB)*100,
    precision_score(y_test,predictedNB,average='weighted')*100,
    recall_score(y_test,predictedNB,average='weighted')*100,
    metrics.accuracy_score(y_test,predictedSVC)*100,
    precision_score(y_test,predictedSVC,average='weighted')*100,
    recall_score(y_test,predictedSVC,average='weighted')*100,
    metrics.accuracy_score(y_test,predictedRF)*100,
    precision_score(y_test,predictedRF,average='weighted')*100,
    recall_score(y_test,predictedRF,average='weighted')*100

]
# this is for plotting purpose
#plt.figure(figsize=(15,15))
print(acc)
index = np.arange(len(label))
plt.bar(index,acc, color=['red','blue','green','red','blue','green','red','blue','green'])
plt.xlabel('Accuracy , Precision & Recall ', fontsize=10)
plt.ylabel('In Percentage', fontsize=10)
plt.xticks(index, label, fontsize=10, rotation=30)
plt.title('Classification Algorithm')
plt.savefig('report.png')
plt.show()