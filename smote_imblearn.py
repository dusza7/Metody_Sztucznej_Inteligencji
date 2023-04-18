import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

#Wczytanie danych
data = pd.read_csv('wdbc.data', header=None)
X = data.iloc[:, 2:]
y = (data.iloc[:, 1] == 'M').astype(int)

#Zainicjowanie walidacji krzyżowej
cross_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=529)

# SMOTE z biblioteki imblearn
smote = SMOTE(random_state=42)
classifier = LGBMClassifier(n_estimators=100)

# Walidacja krzyżowa 5-cio foldowa
scores = []
fold = 0

for train_idx, val_idx in cross_val.split(X, y):
    #Podział na dane treningowe oraz testowe
    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_val = X.iloc[val_idx]
    y_val = y.iloc[val_idx]

    # Zastosowanie SMOTE tylko na zbiorze treningowym
    pipeline = Pipeline([('smote', smote), ('classifier', classifier)])
    pipeline.fit(X_train, y_train)

    prediction = pipeline.predict(X_val)

    balanced_acc_score = balanced_accuracy_score(y_val, prediction)


    print(f"Zrównoważony wynik dokładności fold'u {fold} to: {balanced_acc_score:0.5f}")

    fold += 1
    scores.append(balanced_acc_score)

average_score = np.mean(scores)
print(f'Średnia Zrównoważony wynik dokładności z 5 foldów : {average_score:0.5f}')