
from src.model.logisticsRegression import *
from src.utils.data import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

X,Y = data.dataRandomGenerator(10000,2,2)


print(X.shape)
print(Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)



lr = logiticsRgression()

lr.fit(X_train,y_train,steps=1000)



y_pred = lr.predict(X_test)


print(y_pred[:10])
print(y_test[:10])

r = classification_report(y_pred=y_pred,y_true=y_test)
print(r)


