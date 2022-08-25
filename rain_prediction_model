# Random Forest with SYNOP DATA

from inspect import Parameter
import matplotlib.pylab as plt
import matplotlib.pyplot as pyt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from pandas.plotting import scatter_matrix
from sklearn import metrics
from pandas.plotting import scatter_matrix
from pandas import read_csv
from pandas.plotting import scatter_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.impute import KNNImputer
import pickle

df1 = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/ProjectData2.csv')
df2 = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/imd.csv')
dfs = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/synop00.csv')

df2['DATE'] = df2['YEAR'].apply(str)+' '+df2['MN'].apply(str)+' '+df2['DT'].apply(str)
dfs['DATE'] = dfs['YEAR'].apply(str)+' '+dfs['MN'].apply(str)+' '+dfs['DT'].apply(str)
dfi = pd.merge(df2, dfs, on='DATE', how='inner')
df3 = pd.merge(df1, dfi, on='DATE', how='inner')

param = ['K index', 'Totals totals index', 'Convective Available Potential Energy', 'Showalter index', 'MSLP', 'RH', 'RF_x']
dft = df3[['K index', 'Totals totals index', 'Convective Available Potential Energy', 'Showalter index',  'MSLP', 'RH', 'RF_x']]

df4 = dft
#df4 = df4.dropna()

dfm = df4
#print(dfm['RH'].dtypes)
dfm['RH'] = pd.to_numeric(dfm['RH'], errors='coerce')
#print(dfm['RH'].dtypes)
#print(dfm.isna().sum())
#dfm = dfm.dropna()

Before_imputation = pd.DataFrame(dfm)
imputer = KNNImputer(n_neighbors=10)
After_imputation = imputer.fit_transform(Before_imputation)

#array = dfm.values
array = After_imputation
X = array[:,0:6]
y = array[:,6]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = RandomForestRegressor(n_estimators=1000, random_state=0)
model.fit(X_train, Y_train)

y_pred = model.predict(X_test) 
with open("rfmodel.pkl", 'wb') as file:
    pickle.dump(model, file)

