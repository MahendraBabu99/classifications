from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

data = load_iris()

X = pd.DataFrame( data.data, columns = data.feature_names)
y = pd.Series(data.target, name = "target")

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)

cls = KNeighborsClassifier(n_neighbors=5,algorithm="ball_tree",weights='uniform')
cls.fit(X_train,y_train)

y_pred = cls.predict(X_test)

acc = accuracy_score(y_test,y_pred)
print('accuracy==', acc)