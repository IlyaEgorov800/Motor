import pandas as pd
import numpy as np

dataset = pd.read_csv('Data_80.csv', sep = ';', decimal = ',') # Загружаем train
X_train = dataset.iloc[:, [1,4,5,6,7,8,10,13,14,15,16,17,19,20,21,22,23,24]].values #26 колонок. 6 лишние. Итого 20 
y_train = dataset.iloc[:, [25]].values #26 колонок. 6 лишние. Итого 20 
X_train, y_train = np.array(X_train), np.array(y_train)

from xgboost import XGBClassifier # Запускаем Берлагу (c) Ильф и Петров
regressor = XGBClassifier()
regressor.fit(X_train, y_train)

dataset1 = pd.read_csv('Data_Add_20.csv', sep = ';', decimal = ',') # Загружаем test
X_test = dataset1.iloc[:, [1,4,5,6,7,8,10,13,14,15,16,17,19,20,21,22,23,24]].values #26 колонок. 6 лишние. Итого 20 

prediction = regressor.predict(X_test) # Прогнозируем

df_prediction = pd.DataFrame(prediction)
df = pd.concat([dataset1,df_prediction], axis=1)
df.to_excel("prediction0.xlsx")