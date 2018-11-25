#Шаг 1 моделируем используя только первый датасет
import numpy as np
import pandas as pd

dataset = pd.read_csv('Data_80.csv', sep = ';', decimal = ',') # Загружаем train
Xall = dataset.iloc[:, [0,1,4,5,6,7,8,10,13,14,15,16,17,19,20,21,22,23,24,25]].values #26 колонок. 6 лишние. Итого 20 
X = dataset.iloc[:, [1,4,5,6,7,8,10,13,14,15,16,17,19,20,21,22,23,24]].values
y = dataset.iloc[:, [25]].values

X1 = pd.DataFrame()
y = pd.DataFrame()
zeroline = pd.Series(np.zeros((18)))
for i in range(0, 16138): #Всего 16138 строк. 
    if Xall[i,1] > 25: #10 25 50 100
        y = y.append(pd.Series(Xall[i,19]),ignore_index=True)
        newXline = pd.Series(X[i])
        for j in range(0, 9): #4(90) 9(180) 10(198) Больше - лучше.
            if j >= i:
                newXline = newXline.append(zeroline)
            else:   
                if Xall[i,0] == Xall[i-j-1,0]: #Если название мотора в X[i] == X[i-j-1] - то 
                    newXline = newXline.append(pd.Series(X[i-j-1]))
                else: #Иначе нули
                    newXline = newXline.append(zeroline)
        newXline = pd.DataFrame(newXline).transpose()
        X1 = X1.append(newXline, ignore_index=True)

columnslist = []
for k in range(0, 180): #90 180 198
    columnslist.append(k)  
X1.columns = columnslist

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 180))
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
#df.to_excel("prediction40.xlsx")


#Шаг 2 делаем пайплайн именно для прогноза второго датасета
import numpy as np
import pandas as pd

dataset = pd.read_csv('Data_80.csv', sep = ';', decimal = ',') # Загружаем train
Xall = dataset.iloc[:, [0,1,4,5,6,7,8,10,13,14,15,16,17,19,20,21,22,23,24,25]].values
X = dataset.iloc[:, [1,4,5,6,7,8,10,13,14,15,16,17,19,20,21,22,23,24]].values
y = dataset.iloc[:, [25]].values

X1 = pd.DataFrame()
y = pd.DataFrame()
zeroline = pd.Series(np.zeros((18)))
for i in range(0, 16138): #Всего 16138 строк. 
    if Xall[i,1] > 25: #10 25 50 100
        y = y.append(pd.Series(Xall[i,19]),ignore_index=True)
        newXline = pd.Series(X[i])
        for j in range(0, 9): #4(90) 9(180) 10(198) Больше - лучше.
            if j >= i:
                newXline = newXline.append(zeroline)
            else:   
                if Xall[i,0] == Xall[i-j-1,0]: #Если название мотора в X[i] == X[i-j-1] - то 
                    newXline = newXline.append(pd.Series(X[i-j-1]))
                else: #Иначе нули
                    newXline = newXline.append(zeroline)
        newXline = pd.DataFrame(newXline).transpose()
        X1 = X1.append(newXline, ignore_index=True)

columnslist = []
for k in range(0, 180): #90 180 198
    columnslist.append(k)  
X1.columns = columnslist

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X1)

import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 180))
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y, batch_size = 10, epochs = 100)

dataset = pd.read_csv('Data_Add_20.csv', sep = ';', decimal = ',') # Загружаем test
#X_test = dataset.iloc[:, [1,4,5,6,7,8,10,13,14,15,16,17,19,20,21,22,23,24]].values
Xall = dataset.iloc[:, [0,1,4,5,6,7,8,10,13,14,15,16,17,19,20,21,22,23,24,25]].values
X = dataset.iloc[:, [1,4,5,6,7,8,10,13,14,15,16,17,19,20,21,22,23,24]].values
y = dataset.iloc[:, [25]].values

X1 = pd.DataFrame()
y = pd.DataFrame()
zeroline = pd.Series(np.zeros((18)))
for i in range(0, 4051): #Всего 16138 строк. 
    if Xall[i,1] > 25: #10 25 50 100
        y = y.append(pd.Series(Xall[i,19]),ignore_index=True)
        newXline = pd.Series(X[i])
        for j in range(0, 9): #4(90) 9(180) 10(198) Больше - лучше.
            if j >= i:
                newXline = newXline.append(zeroline)
            else:   
                if Xall[i,0] == Xall[i-j-1,0]: #Если название мотора в X[i] == X[i-j-1] - то 
                    newXline = newXline.append(pd.Series(X[i-j-1]))
                else: #Иначе нули
                    newXline = newXline.append(zeroline)
        newXline = pd.DataFrame(newXline).transpose()
        X1 = X1.append(newXline, ignore_index=True)
X1.columns = columnslist
X_test = sc.transform(X1)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

df_prediction = pd.DataFrame(y_pred)
df = pd.concat([dataset,df_prediction], axis=1)
df.to_excel("prediction4.xlsx")