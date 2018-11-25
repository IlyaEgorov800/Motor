#1
import pandas as pd
#import numpy as np

dataset = pd.read_csv('Data_80.csv', sep = ';', decimal = ',') # Загружаем train
X = dataset.iloc[:, [4,5,6,7,8,10,13,14,15,16,17,19,20,21,22,23,24]].values #26 колонок. 6 лишние. Итого 20 
y = dataset.iloc[:, [25]].values
#X_train, y_train = np.array(X_train), np.array(y_train)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from xgboost import XGBRegressor
regressor = XGBRegressor()
#regressor = XGBRegressor(max_depth = 10, n_estimators = 500) 
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test) # Прогнозируем
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#2
dataset = pd.read_csv('Data_80.csv', sep = ';', decimal = ',')
X_train = dataset.iloc[:, [4,5,6,7,8,10,13,14,15,16,17,19,20,21,22,23,24]].values
y_train = dataset.iloc[:, [25]].values

from xgboost import XGBRegressor
regressor = XGBRegressor()
#regressor = XGBRegressor(max_depth = 10, n_estimators = 500) 
regressor.fit(X_train, y_train)

dataset1 = pd.read_csv('Data_Add_20.csv', sep = ';', decimal = ',')
X_test = dataset1.iloc[:, [4,5,6,7,8,10,13,14,15,16,17,19,20,21,22,23,24]].values

prediction = regressor.predict(X_test) # Прогнозируем
prediction = (prediction > 0.5)

df_prediction = pd.DataFrame(prediction)
df = pd.concat([dataset1,df_prediction], axis=1)
df.to_excel("prediction5.xlsx")