# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report ,confusion_matrix 

col_names =['clump_thickness','uniformity_of_cell_size','uniformity_of_cell_shape','marginal_adhesion','single_epithelial_cell_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitosis','label']
pima=pd.read_csv("farag.csv",header=None,names=col_names)
pima.head()
#features selection
x = pima.iloc[:, :-1].values
y = pima.iloc[:, 9].values



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)


scaler=StandardScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#Building KNN Model
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train,y_train)
y_pred =classifier.predict(x_test)

#Evaluating Model
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

error = []
# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))
    
    #Visualizing
plt.figure(figsize=(12,6))
plt.plot(range(1,40), error,color='red',linestyle='dashed',marker='o',markerfacecolor='blue',markersize=10)
plt.title('Error rate K value ')
plt.xlabel('K value')
plt.ylabel('Mean Error')
