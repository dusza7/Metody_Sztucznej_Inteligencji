import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from lightgbm import LGBMClassifier
from adasyn import Adasyn
from imblearn.over_sampling import SMOTE


Methods = [
    {'our_adasyn': Adasyn(beta=0.9, n_neighbors=3, random_state=42)},
    {'imblearn_adasyn': ADASYN(random_state=42)},
    {'imblearn_smote': SMOTE(random_state=42)},
    {'imblearn_random': RandomOverSampler(random_state=42)}
]

#Wczytanie danych
data = pd.read_csv('wdbc.data', header=None)
X = data.iloc[:, 2:]
y = (data.iloc[:, 1] == 'M').astype(int)

# Plotting before oversampling
plt.figure(figsize=(10, 6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y)
plt.title('Before Oversampling')
plt.show()

#Zainicjowanie walidacji krzyżowej
cross_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=529)

num_folds_repeats = 5
num_methods = 4

# utworzenie macierzy, w której zapiszemy wyniki foldów dla danej metody
results = np.zeros((num_folds_repeats, num_methods))


for method_idx, method_dict in enumerate(Methods):
    for name, method in method_dict.items():
        # ADASYN z biblioteki imblearn
        print(name)
        current_method = method
        classifier = LGBMClassifier(n_estimators=100)

        scores = []
        fold = 0

        for idx, (train_idx, val_idx) in enumerate(cross_val.split(X, y)):
            #Podział na dane treningowe oraz testowe
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]

            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            pipeline = Pipeline([('method', current_method), ('classifier', classifier)])
            pipeline.fit(X_train, y_train)

            # w celu wygenerowania wykresów
            sampler = pipeline.named_steps['method']
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

            # wykresy tylko dla pierwszych foldów
            if fold == 0:
                plt.figure(figsize=(10, 6))
                if name == "our_adasyn":
                    plt.scatter(X_resampled[:, 0], X_resampled[:, 1], c=y_resampled)
                else: #numpy array
                    plt.scatter(X_resampled.iloc[:, 0], X_resampled.iloc[:, 1], c=y_resampled)
                plt.title(f'After Oversampling with {name}')
                plt.show()

            prediction = pipeline.predict(X_val)

            balanced_acc_score = balanced_accuracy_score(y_val, prediction)
            results[idx, method_idx] = balanced_acc_score

            print(f"Zrównoważony wynik dokładności fold'u {fold} to: {balanced_acc_score:0.5f}")

            fold += 1
            scores.append(balanced_acc_score)

        average_score = np.mean(scores)
        print(f'Średnia Zrównoważony wynik dokładności z 5 foldów dla metody ' + name + f': {average_score:0.5f}')

np.savetxt('wyniki.csv', results, delimiter=',', fmt='%.5f')
