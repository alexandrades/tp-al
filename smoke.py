import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, f_classif
import statistics

data = pd.read_csv('dataset.csv', sep=',')

X = data.drop(columns='SMK_stat_type_cd', axis=1)
y = data['SMK_stat_type_cd']

## LABEL ENCODING
label_encoder = LabelEncoder()
X['sex'] = label_encoder.fit_transform(X['sex'])
X['DRK_YN'] = label_encoder.fit_transform(X['DRK_YN'])

X, x_test, y, y_test = train_test_split(X, y, test_size=.2, stratify=y)

## MAPA DE CALOR CORRELAÇÃO
# plt.figure(figsize=(10, 7))
# sns.heatmap(X.corr(), annot = True, fmt = '.2f', cmap='Blues')
# plt.title('Correlação entre variáveis do dataset de Iris')
# plt.show()

f1_macro = []

for i in range(2,24):
    X_temp = SelectKBest(f_classif, k=i).fit_transform(X, y)
    model = RandomForestClassifier()
    x1, x2, y1, y2 = train_test_split(X_temp, y, test_size=0.2, stratify=y)
    # scores = cross_val_score(model, X_temp, y, cv=5, scoring='f1_macro')
    model.fit(x1, y1)
    with open('log.txt', 'a') as log:
        log.write(f'{i} features:\n\tacuracia: {model.score(x2, y2)}\n\t')
