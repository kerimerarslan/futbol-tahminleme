import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv("ypyzeka_veriseti.csv")
data.info()
data.head()

data.drop(["Evsahibi","Deplasman","EVG","DEPG"], axis=1,inplace = True)
x=data.drop(["MS"],axis=1)
y = data["MS"]
sns.distplot(y)
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as cm




x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(x_train, y_train)

predictions= classifier.predict(x_test)

errors = abs(predictions - y_test)
print('Ortalama Hata:', round(np.mean(errors), 2), 'unit.')
mape = 100* (errors / y_test)
basari = 100- np.mean(mape)
print('Başarı Yüzdesi:', abs(round(basari,3)), '%.')




xnew = [[2,1,1,7,6,2,3,12,15,4,4,3,3,0,0]]
ynew = classifier.predict(xnew)
print("X=%s, Tahmin=%s" % (xnew[0], ynew[0]))








#from ipywidgets import Image
#from io import StringIO
#import pydotplus
#from sklearn.tree import export_graphviz
#
#dot_data = StringIO()
#export_graphviz(d_tree1, feature_names = x.columns,
#               out_file = dot_data, filled = True)
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#Image(value = graph.create_png())



#classifier2 = GaussianNB()
#classifier2.fit(x_train, y_train)
#
#predictions1= classifier2.predict(x_test)
#
#errors = abs(predictions1 - y_test)
#print('Mean Absolute Error:', round(np.mean(errors), 2), 'unit.')
#mape = 100* (errors / y_test)
#accuracy = 100- np.mean(mape)
#print('Accuracy:', abs(round(accuracy,3)), '%.')



#plt.figure(figsize=(16, 9))

#ranking = classifier.feature_importances_
#features = np.argsort(ranking)[::-1][:10]
#columns = x.columns
#
#plt.title("Feature importances based on Decision Tree Regressor", y = 1.03, size = 18)
#plt.bar(range(len(features)), ranking[features], color="lime", align="center")
#plt.xticks(range(len(features)), columns[features], rotation=80)
#plt.show()

#accuracy = tf.metrics.accuracy(y_train, y_test)
#print(accuracy)
#




