import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
def categorical_to_binary_solution(data, columns):
    for column in columns:
        f=True if column=='Fjob' else 0
        m = True if column == 'Mjob' else 0
        categories = list(data[column].unique())
        res = {}
        for el in categories:
            res[el] = []
        for i in range(len(data[column])):
            val = data[column].iloc[i]
            for el in res:
                res[el].append(int(val == el))
        for cat in res:
            pref=''
            if f:
                pref='f_'
            elif m:
                pref='m'
            data[pref+cat] = res[cat]
    return data
data = pd.read_csv('student-mat.csv')
data = data[data['G3'] != 0] #выброс
X = data.drop(columns=['G3'])
y = data['G3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, random_state=0)
#данных мало 357, можно k-Fold или просто небольшую тест.часть
