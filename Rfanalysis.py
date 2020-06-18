# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 16:57:12 2019

@author: gaura
"""


import pandas as pd
from sklearn.preprocessing import LabelEncoder
df1 = pd.read_csv("indiadata.csv",engine='python',usecols = ['gname'])
df = pd.read_csv("indiadata.csv",skipinitialspace=True, usecols = ['city','longitude','latitude','attacktype1_txt','targtype1_txt'])

import pandas as pd
# read in data to use for plotted points
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
bf = bf.iloc[:,:].values
bf1 = bf1.iloc[:,:].values



from sklearn.model_selection import train_test_split
X_train , X_test ,y_train,y_test = train_test_split(bf,bf1,test_size = 0.3 , random_state = 0 )
from sklearn.neighbors import KNeighborsClassifier
accuracy=[]
pre=[]
recall=[]
maxacc = 0
best=0 
for i in range(2,50):
  
  from sklearn.ensemble import RandomForestClassifier
  forestclass=RandomForestClassifier(n_estimators=i)
#Train the model using the training sets y_pred=clf.predict(X_test)
  forestclass.fit(X_train,y_train)
  predicted=forestclass.predict(X_test)
  from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
  accuracy.append(metrics.accuracy_score(y_test,predicted)*100)
  temp = metrics.accuracy_score(y_test,predicted)*100
  if temp  > maxacc :
      maxacc=temp
      best=i
  from sklearn.metrics import precision_score
  pre.append(precision_score(y_test,predicted, average='weighted')*100)
  from sklearn.metrics import recall_score
  recall.append(recall_score(y_test,predicted,average='weighted')*100)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
# Data
pldf=pd.DataFrame({'x': range(2,50)})
pldf['Accuracy'] =accuracy
pldf['Precision'] =pre
pldf['Recall'] =recall 
# multiple line plot
plt.title('Random forest classification Analysis')
plt.xlabel('n estimators', fontsize=10)
plt.ylabel('In Percentage', fontsize=10)
    
plt.plot( 'x', 'Accuracy', data=pldf, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.plot( 'x', 'Precision', data=pldf, marker='', color='olive', linewidth=2)
plt.plot( 'x', 'Recall', data=pldf, marker='', color='olive', linewidth=2, linestyle='dashed', label="Recall")

plt.savefig('report2.png')

plt.legend()
