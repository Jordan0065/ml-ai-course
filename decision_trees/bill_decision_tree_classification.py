import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix


dataset = pd.read_csv("bill_authentication.csv")

print(dataset.shape)

dataset.head()

#preparing the dataset
X = dataset.drop('Class', axis = 1)
y = dataset['Class']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.20)


classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Decision tree regression

