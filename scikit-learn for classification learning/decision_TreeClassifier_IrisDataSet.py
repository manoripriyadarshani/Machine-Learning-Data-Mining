from sklearn import tree
from sklearn.datasets import load_iris
from sklearn import tree
import os
import pydotplus
from IPython.display import Image
from PIL import Image , ImageDraw


#Using the Iris dataset construct a tree as follows:
iris = load_iris ()
clf = tree. DecisionTreeClassifier ()
clf = clf.fit(iris.data , iris.target)

with open("iris.dot", 'w') as f:
	f = tree. export_graphviz (clf , out_file=f)
	
os.unlink('iris.dot')

#generate a PDF file and png
dot_data = tree.export_graphviz (clf , out_file =None)
graph = pydotplus . graph_from_dot_data ( dot_data )
graph.write_pdf ("iris.pdf")
graph.write_png ("iris.png")


#including colors to nodes
dot_data = tree.export_graphviz (clf , out_file =None ,feature_names =iris.feature_names ,class_names =iris.target_names ,filled=True , rounded=True ,special_characters =True)
graph = pydotplus.graph_from_dot_data( dot_data )
graph.write_png("iris.png")
image = Image.open("iris.png")
image.show ()

#model is used to predict the class of samples
predicted=clf.predict(iris.data [:1, :])
print(predicted)

#probability of each class c predicted
predicted_proba=clf.predict_proba(iris.data [:1, :])
print(predicted_proba)






