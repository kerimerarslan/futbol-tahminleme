import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

data=pd.read_csv("ypyzeka_veriseti.csv")
data.info()
data.head()

data.drop(["Evsahibi","Deplasman","EVG","DEPG"], axis=1,inplace = True)
x=data.drop(["MS"],axis=1)
y = data["MS"]
#sns.distplot(y)
#plt.show()



from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as cm




x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
d_tree1 = DecisionTreeRegressor(max_depth = 3, random_state=0)
d_tree1.fit(x_train, y_train)

predictions = d_tree1.predict(x_test)
errors = abs(predictions - y_test)
print('Ortalama Hata:', round(np.mean(errors), 2), 'unit.')
mape = 100* (errors / y_test)
basari = 100- np.mean(mape)
print('Başarı Yüzdesi:', round(basari,3), '%.')

#accuracy_score(y_test,predictions)

#from ipywidgets import Image
#from io import StringIO
#import pydotplus
#from sklearn.tree import export_graphviz
#
#dot_data = StringIO()
#export_graphviz(d_tree1, feature_names = x.columns,
#               out_file = dot_data, filled = True)
#graph = t
#Image(value = graph.create_png())



d_tree2 = DecisionTreeRegressor(max_depth = 8, random_state=42)
d_tree2.fit(x_train, y_train)
predictions1 = d_tree2.predict(x_test)

errors = abs(predictions1 - y_test)
print('Ortalama Hata:', round(np.mean(errors), 3), 'unit.')
mape = 100* (errors / y_test)
basari = 100-np.mean(mape)
print('Başarı Yüzdesi:', abs(round(basari,3)), '%.')




plt.figure(figsize=(16, 9))

ranking = d_tree2.feature_importances_
features = np.argsort(ranking)[::-1][:10]
columns = x.columns

plt.title("Karar Ağacı", y = 1.03, size = 18)
plt.bar(range(len(features)), ranking[features], color="lime", align="center")
plt.xticks(range(len(features)), columns[features], rotation=80)
plt.show()



print("İEG:")
ieg=input()
print("İDG:")
idg=input()
print("İS:")
iso=input()
print("ES:")
es=input()
print("DS:")
ds=input()
print("EKS:")
eks=input()
print("DKS:")
dks=input()
print("EF:")
ef=input()
print("DF:")
df=input()
print("EK:")
ek=input()
print("DK:")
dk=input()
print("ESK:")
esk=input()
print("DSK:")
dsk=input()
print("EKK:")
ekk=input()
print("DKK:")
dkk=input()





xnew = [[ieg,idg,iso,es,ds,eks,dks,ef,df,ek,dk,esk,dsk,ekk,dkk]]
ynew = d_tree1.predict(xnew)
print("X=%s, Predicted=%s" % (xnew[0], ynew[0]))



