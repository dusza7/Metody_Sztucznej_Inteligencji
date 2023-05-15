import numpy as np
from sklearn import neighbors


class Adasyn:
#dopisac dziedziczenie po klasach
    def __init__(self, beta, n_neighbors, random_state=42):
        self.beta = beta
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit(self, X, y):
        self.X_, self.y_ = X, y

        # Obliczanie liczby przykładów w klasie mniejszościowej (minor_class) oraz w klasie większościowej(major_class)
        self.minor_class_ = int(sum(y))
        self.major_class_ = len(y) - self.minor_class_

        # Klasyfikator n_neighbors-najbliższych sąsiadów musi być dopasowany do danych
        self.clf_ = neighbors.KNeighborsClassifier(n_neighbors=self.n_neighbors)
        self.clf_.fit(X, y)

        #Obliczenie Ri (stopień nierówności między klasami) oraz Minority_per_xi (Jest to lista z indeksami elementów z mniejszościowej klasy, które są najbliższymi sąsiadami każdego elementu w klasie mniejszościowej)
        self.Ri_, self.Minority_per_xi_ = self._calculate_Ri()

        # Ile przykładów do wygeerowania
        self.G_ = int((self.major_class_ - self.minor_class_) * self.beta)

        # Znormalizowanie Ri aby określić wagę przykładu klasy mniejszościowej przy generowaniu sztucznych przykładów.
        Rhat_i = self.Ri_ / np.sum(self.Ri_)
        assert np.isclose(np.sum(Rhat_i), 1.0, rtol=1e-05, atol=1e-08)

        # Obliczenie ilości danych do wygenerowania dla każdego przykładu klasy mniejszościowej
        self.Gi_ = np.round(Rhat_i * self.G_).astype(int)

        # Generowanie przykładów syntetycznych
        syn_data = []
        for i in range(self.minor_class_):
            xi = X.iloc[i, :].to_numpy().reshape(1, -1)
            if self.Gi_[i] == 0:
                continue
            for j in range(self.Gi_[i]):
                if len(self.Minority_per_xi_[i]) == 0:
                    continue
                index = np.random.choice(self.Minority_per_xi_[i])
                xzi = X.iloc[index, :].to_numpy().reshape(1, -1)
                si = xi + (xzi - xi) * np.random.uniform(0, 1)
                syn_data.append(si)

        self.synthetic_X_ = np.concatenate((X, np.concatenate(syn_data, axis=0)), axis=0)
        self.synthetic_y_ = np.concatenate((y, np.ones(len(syn_data))), axis=0)
        return self

    def fit_resample(self, X, y):
        return self.fit(X, y).synthetic_X_, self.fit(X, y).synthetic_y_

    def _calculate_Ri(self):
        Ri = []
        Minority_per_xi = []
        for i in range(self.minor_class_):
            xi = self.X_.iloc[i, :].to_numpy().reshape(1, -1)
            neighbours = self.clf_.kneighbors(xi, self.n_neighbors+1, False)[0][1:]
            count = 0
            minority = []
            for value in neighbours:
                if self.y_.iloc[value] != self.y_.iloc[i]:
                    count += 1
                    minority.append(value)
            Ri.append(count / self.n_neighbors)
            Minority_per_xi.append(minority)
        return np.array(Ri), Minority_per_xi

