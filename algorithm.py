import pandas as pd
import numpy as np
import os

data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dataset_number_6.csv'))

F_star = np.inf
columns = data.columns
theta_star = 0
j_star = 0

for j in range(len(columns)):
    if 'Feature' in columns[j]:
        data.sort_values(by=columns[j])
        F = data.loc[data['Label'] == 1, columns[j]].sum()

        if F < F_star:
            F_star = F
            theta_star = data[columns[j]].iloc[0] - 1
            j_star = j
        for i in range(data.shape[0]):
            F = F - data['Label'].iloc[i] * data['Distribution'].iloc[i]
            if F < F_star and i+1<data.shape[0] and data[columns[j]].iloc[i] != data[columns[j]].iloc[i+1]:
                F_star = F
                theta_star = 0.5 * (data[columns[j]].iloc[i] + data[columns[j]].iloc[i+1])
                j_star = j
            elif F < F_star and i+1 == data.shape[0]:
                F_star = F
                theta_star = 0.5 * (data[columns[j]].iloc[i] + data[columns[j]].iloc[i] + 1)
                j_star = j

print(j_star)
print(theta_star)
