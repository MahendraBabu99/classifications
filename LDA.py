from sklearn.datasets import load_iris
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

regressor = LinearDiscriminantAnalysis(solver='eigen').fit(X_train,y_train)

y_pred = regressor.predict(X_test)

print("acc_score=", accuracy_score(y_test,y_pred))