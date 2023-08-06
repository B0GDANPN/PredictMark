import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
def categorical_to_binary_solution(data:pd.DataFrame, columns):
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
                pref='m_'
            data[pref+cat] = res[cat]
    data.drop(columns,axis=1,inplace=True)
data = pd.read_csv('student-mat.csv')
data = data[data['G3'] != 0] #выброс
X = data.drop(columns=['G3'])
y = data['G3']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, random_state=0)
#векторизировал получил значения с которыми G3 коррелириует >=0.15
nessecary_column=[
'age',
'Medu',
'Fedu',
'higher',
'failures',
'studytime',
'schoolsup',
'absences',
'goout',
'f_teacher',
'm_health',
'G1'
]
categorical_to_binary_solution(X_train, ['Fjob','Mjob','guardian','reason'])
X_train['school'].replace('GP',1,inplace=True)
X_train['school'].replace('MS',0,inplace=True)
X_train['address'].replace('U',1,inplace=True)
X_train['address'].replace('R',0,inplace=True)

X_train['gender'].replace('M',1,inplace=True)
X_train['gender'].replace('F',0,inplace=True)

X_train['famsize'].replace('GT3',1,inplace=True)
X_train['famsize'].replace('LE3',0,inplace=True)

X_train['Pstatus'].replace('T',1,inplace=True)
X_train['Pstatus'].replace('A',0,inplace=True)

X_train['schoolsup'].replace('yes',1,inplace=True)
X_train['schoolsup'].replace('no',0,inplace=True)

X_train['paid'].replace('yes',1,inplace=True)
X_train['paid'].replace('no',0,inplace=True)

X_train['activities'].replace('yes',1,inplace=True)
X_train['activities'].replace('no',0,inplace=True)

X_train['nursery'].replace('yes',1,inplace=True)
X_train['nursery'].replace('no',0,inplace=True)

X_train['higher'].replace('yes',1,inplace=True)
X_train['higher'].replace('no',0,inplace=True)

X_train['internet'].replace('yes',1,inplace=True)
X_train['internet'].replace('no',0,inplace=True)

X_train['romantic'].replace('yes',1,inplace=True)
X_train['romantic'].replace('no',0,inplace=True)

X_train['famsup'].replace('yes',1,inplace=True)
X_train['famsup'].replace('no',0,inplace=True)
for col in X_train.columns:
    if not col in nessecary_column:
        X_train.drop(col,inplace=True,axis=1)
#нормирую X
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
model = LinearRegression()
model.fit(X_train_scaled, y_train)
ans=model.score(X_train_scaled, y_train)
#данных мало 357, можно k-Fold или просто небольшую тест.часть
print(ans)
