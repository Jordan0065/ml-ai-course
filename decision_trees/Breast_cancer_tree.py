from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


#importing the breast cancer dataset
breast_cancer = load_breast_cancer()


#creating a decision tree
#depth = levels in the tree
clf = DecisionTreeClassifier(max_depth = 3)

#fitting the data to match the model
clf.fit(breast_cancer.data,breast_cancer.target)


#plotting the decision tree
plt.figure(figsize = (25,10))
a = plot_tree(clf,feature_names = breast_cancer.feature_names, 
              class_names = breast_cancer.target_names,
               filled = True,
                rounded = True,
                 fontsize = 14)

plt.show()


