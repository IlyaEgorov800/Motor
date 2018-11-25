#1
import pandas as pd
import numpy as np

dataset = pd.read_csv('Data_80.csv', sep = ';', decimal = ',') # Загружаем train
X_train = dataset.iloc[:, [1,4,5,6,7,8,10,13,14,15,16,17,19,20,21,22,23,24]].values #26 колонок. 6 лишние. Итого 20 
y_train = dataset.iloc[:, [25]].values #26 колонок. 6 лишние. Итого 20 
X_train, y_train = np.array(X_train), np.array(y_train)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 18))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

y_pred = classifier.predict(X_test) # Прогнозируем
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#dataset1 = pd.read_csv('Data_Add_20.csv', sep = ';', decimal = ',') # Загружаем test
#X_test = dataset1.iloc[:, [1,4,5,6,7,8,10,13,14,15,16,17,19,20,21,22,23,24]].values #26 колонок. 6 лишние. Итого 20 
#X_test = sc.transform(X_test)
#
#y_pred = classifier.predict(X_test) # Прогнозируем
#y_pred = (y_pred > 0.5)
#
#df_prediction = pd.DataFrame(y_pred)
#df = pd.concat([dataset1,df_prediction], axis=1)
#df.to_excel("prediction30.xlsx")

#2
import pandas as pd
import numpy as np

dataset = pd.read_csv('Data_80.csv', sep = ';', decimal = ',') # Загружаем train
X_train = dataset.iloc[:, [1,4,5,6,7,8,10,13,14,15,16,17,19,20,21,22,23,24]].values #26 колонок. 6 лишние. Итого 20 
y_train = dataset.iloc[:, [25]].values #26 колонок. 6 лишние. Итого 20 
X_train, y_train = np.array(X_train), np.array(y_train)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 18))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

dataset1 = pd.read_csv('Data_Add_20.csv', sep = ';', decimal = ',') # Загружаем test
X_test = dataset1.iloc[:, [1,4,5,6,7,8,10,13,14,15,16,17,19,20,21,22,23,24]].values #26 колонок. 6 лишние. Итого 20 
X_test = sc.transform(X_test)

y_pred = classifier.predict(X_test) # Прогнозируем
y_pred = (y_pred > 0.5)

df_prediction = pd.DataFrame(y_pred)
df = pd.concat([dataset1,df_prediction], axis=1)
df.to_excel("prediction3.xlsx")