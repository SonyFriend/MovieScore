#Original data size is 45456,but we interested the movie which is made in USA. 
#So the data size became 20920. We remove missing value from 'budget' and 'revenue', the data size became 4381.
#We want to predict the movie will have sequel or not.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import math
df=open('C:/Users/user/Downloads/pythonCode/movie_4381.csv')
movie=pd.read_csv(df)
series_row=movie.iloc[:,1]
series=[]
#If the movie have sequel,representing 1,otherwise 0.
for i in range(0,len(movie)):
	if int(isinstance(series_row[i],float))==1:
		series.append(0)
	else:
		series.append(1)
movie['series']=series#which is our response(Y)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
accurate=[]
for i in range(20):
	#We split the data into training set and test set,80% to be training,20% to be test.
	x_train,x_test,y_train,y_test=train_test_split(movie[['budget','revenue','popularity','runtime']],movie['series'],test_size=0.2)
	sc=StandardScaler()
	sc.fit(x_train.astype(float))
	x_train_std=sc.transform(x_train.astype(float))
	x_test_std=sc.transform(x_test.astype(float))
	svm=SVC(kernel='linear',probability=True)
	svm.fit(x_train_std,y_train.values)
	#print(metrics.classification_report(y_test,svm.predict(x_test_std)))
	mat=metrics.confusion_matrix(y_test,svm.predict(x_test_std))
	accurate.append((mat[0,0]+mat[1,1])/sum(sum(mat)))
print(svm.predict(x_test_std)) #which is the result for prediction
print(np.average(accurate))#which is accuracy
