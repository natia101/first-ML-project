import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Carga de base de datos
url='https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/'+\
    'main/bank-marketing-campaign-data.csv'
df=pd.read_csv(url,sep=';')

# Transformacion de variables
# 'object' a 'category'
df[df.select_dtypes('object').columns]=df[df.select_dtypes('object').columns].astype('category')
# 'pdays' a binaria yes/no
df['pdays']=['no' if i==999 else 'yes' for i in df['pdays']]
df['pdays']=df['pdays'].astype('category')

# Eliminacion de duplicados
df=df.drop_duplicates()

# Remplazo de 'unknown' por la moda para variables categoricas
uk=[]
for i in df:
    [uk.append(i) for n in df[i].unique() if n=='unknown']

for i in uk:
    df=df.replace({i:'unknown'}, df[i].mode()[0]) 

# Recategorizacion de la variable 'education'
df=df.replace({'education':['basic.9y','basic.6y','basic.4y']}, 'middle_school')

# Divicion de la variable 'age' por rangos de edad
df['age_rang']=pd.cut(df['age'],bins=[10,20,30,40,50,60,70,80,90,100])
df['age_rang']=df['age_rang'].astype('category')

# Encoding
# Hot encoding
df_cod=pd.get_dummies(df,columns=df.select_dtypes('category').drop(columns=['y']).columns)
# Ordinal/Label encoding
df_cod['y']=df.y.cat.codes

# Seleccion de features
df_cod=df_cod.drop(columns=['euribor3m', 'nr.employed'])

f=[]
pd.set_option('display.max_rows',110)
for i,r in df_cod.corr()[['y']].iterrows():
    if (r[0] > 0.05) | (r[0] < -0.05):
        f.append(i)

X=df_cod[f].drop(columns='y')
y=df_cod['y']

# Seleccion de muestra de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=57)

# Estandarizacion de las variables
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform (X_test)

# Definicion del Modelo
w={0:11,1:89} #Pesos
lg= LogisticRegression(random_state=57, class_weight=w)

lg.fit(X_train_sc,y_train) # Se entrena

y_pred = lg.predict(X_test_sc) # Se testea

# Se guarda el modelo
import pickle
filename = '/workspace/first-ML-project/models/finalized_model.sav'
pickle.dump(lg, open(filename, 'wb'))