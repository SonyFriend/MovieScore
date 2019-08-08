#We use decision tree to predict the movie's score
#We use regression tree to establish it because our response is a continuous variable

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import csv

df = open('C:/Users/user/Downloads/pythonCode/movie_4381.csv')

movie=pd.read_csv(df)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
#We split the data into training set and test set,70% to be training set,20% to be test set
x_train,x_test,y_train,y_test=train_test_split(movie[['runtime','revenue','budget','popularity']],movie[['vote_average']],test_size=0.3,random_state=0)
tree=DecisionTreeRegressor(criterion='mse',max_depth=2,random_state=0)
tree=tree.fit(x_train,y_train)

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
import pydotplus 
#Watch out! If your Graphviz report "GraphViz’s executables not found"
#You should add this path:'C:/Program Files (x86)/Graphviz2.38/bin/' to your 'PATH'
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
#visualization
with open("tree.dot", 'w') as f:
 f = export_graphviz(tree, out_file=f)
 dot_data=export_graphviz(tree,out_file=None,feature_names=['runtime','revenue','budget','popularity'],filled=True,rounded=True,special_characters=True)

y=y_test['vote_average']
predict=tree.predict(x_test)

graph=pydotplus.graph_from_dot_data(dot_data)

graph.write_pdf("tree.pdf") 
from sklearn import metrics
#We use Mean Absolute Error to determine the prediction is good or not because our response is continuous
#The prediction is pretty good if MAE is less than 0.2 
print(np.mean(abs(np.multiply(np.array(y_test.T-predict),np.array(1/y_test))))) #MAE=0.1071
