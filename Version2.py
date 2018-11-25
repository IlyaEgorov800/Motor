#Шаг 1 моделируем используя только первый датасет
import numpy as np
import pandas as pd

dataset = pd.read_csv('Data_80.csv', sep = ';', decimal = ',') # Загружаем train
Xall = dataset.iloc[:, [0,1,4,5,6,7,8,10,13,14,15,16,17,19,20,21,22,23,24,25]].values #26 колонок. 6 лишние. Итого 20 
X = dataset.iloc[:, [1,4,5,6,7,8,10,13,14,15,16,17,19,20,21,22,23,24]].values
#y = dataset.iloc[:, [25]].values

X1 = pd.DataFrame()
y = pd.DataFrame()
zeroline = pd.Series(np.zeros((18)))
for i in range(0, 16138): #Всего 16138 строк. 
    if Xall[i,1] > 50: 
        y = y.append(pd.Series(Xall[i,19]),ignore_index=True)
        
        newXline = pd.Series(X[i]) #Не так - надо брать вдва правых столбца из предыдущей(!!!) строки
        
        for j in range(0, 20): #Больше - лучше. У меня столько, изза малой мощности машины
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
for k in range(0, 378):#198
    columnslist.append(k)  
X1.columns = columnslist

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.2, random_state = 0)

from xgboost import XGBClassifier # Запускаем Берлагу (c) Ильф и Петров
#regressor = XGBClassifier() 
regressor = XGBClassifier(max_depth = 12, n_estimators = 750) 
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#Конец Шаг 1 моделируем используя только первый датасет


#Шаг 2 делаем пайплайн именно и только для прогноза второго датасета
import numpy as np
import pandas as pd

dataset = pd.read_csv('Data_80.csv', sep = ';', decimal = ',') # Загружаем train
Xall = dataset.iloc[:, [0,1,4,5,6,7,8,10,13,14,15,16,17,19,20,21,22,23,24]].values #26 колонок. 6 лишние. Итого 20 
X = dataset.iloc[:, [1,4,5,6,7,8,10,13,14,15,16,17,19,20,21,22,23,24]].values
y = dataset.iloc[:, [25]].values

X1 = pd.DataFrame()
zeroline = pd.Series(np.zeros((18)))
for i in range(0, 16138): #Всего 16138 строк. 
    newXline = pd.Series(X[i])
    for j in range(0, 10): #Больше - лучше. У меня столько, изза малой мощности машины
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
for k in range(0, 198):
    columnslist.append(k)  
X1.columns = columnslist

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size = 0.2, random_state = 0)

from xgboost import XGBClassifier # Запускаем Берлагу (c) Ильф и Петров
#regressor = XGBClassifier() 
regressor = XGBClassifier(max_depth = 12, n_estimators = 750) 
regressor.fit(X1, y)

dataset1 = pd.read_csv('Data_Add_20.csv', sep = ';', decimal = ',') # Загружаем train
Xall3 = dataset1.iloc[:, [0,1,4,5,6,7,8,10,13,14,15,16,17,19,20,21,22,23,24]].values #26 колонок. 6 лишние. Итого 20 
X3 = dataset1.iloc[:, [1,4,5,6,7,8,10,13,14,15,16,17,19,20,21,22,23,24]].values 
X4 = pd.DataFrame()
zeroline = pd.Series(np.zeros((18)))
for i in range(0, 4051): #Всего 4051 строк. 
    newXline = pd.Series(X3[i])
    for j in range(0, 10): #Больше - лучше. У меня столько, изза малой мощности машины
        if j >= i:
            newXline = newXline.append(zeroline)
        else:   
            if Xall3[i,0] == Xall3[i-j-1,0]: #Если название мотора в X[i] == X[i-j-1] - то 
                newXline = newXline.append(pd.Series(X3[i-j-1]))
            else: #Иначе нули
                newXline = newXline.append(zeroline)
    newXline = pd.DataFrame(newXline).transpose()
    X4 = X4.append(newXline, ignore_index=True)

X4.columns = columnslist

prediction = regressor.predict(X4) # Прогнозируем

df_prediction = pd.DataFrame(prediction)
df = pd.concat([dataset1,df_prediction], axis=1)
df.to_excel("prediction2.xlsx")
#Конец Шаг 2