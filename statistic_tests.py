import numpy as np
from scipy.stats import ttest_rel

# wczytanie pliki
results = np.load('wyniki.npy')

# inicjalizacja macierzy
num_classifiers = results.shape[1]
t_statistic_matrix = np.zeros((num_classifiers, num_classifiers))
p_value_matrix = np.zeros((num_classifiers, num_classifiers))
advantage_matrix = np.zeros((num_classifiers, num_classifiers), dtype=bool)

for i in range(num_classifiers):
    for j in range(num_classifiers):
            t_statistic, p_value = ttest_rel(results[:, i], results[:, j])
            t_statistic_matrix[i, j] = t_statistic
            p_value_matrix[i, j] = p_value
            advantage_matrix[i, j] = np.mean(results[:, i]) > np.mean(results[:, j])

alpha = 0.05
significant_advantage_matrix = (p_value_matrix < alpha)
statistical_advantage_matrix = advantage_matrix * significant_advantage_matrix

np.savetxt('t_statistic_matrix.csv', t_statistic_matrix, delimiter=',', fmt='%.5f')
np.savetxt('p_value_matrix.csv', p_value_matrix, delimiter=',', fmt='%.5f')
np.savetxt('advantage_matrix.csv', advantage_matrix, delimiter=',', fmt='%d')
np.savetxt('significant_advantage_matrix.csv', significant_advantage_matrix, delimiter=',', fmt='%d')
np.savetxt('statistical_advantage_matrix.csv', statistical_advantage_matrix, delimiter=',', fmt='%d')

print("Macierze zostały zapisane do plików .csv")
