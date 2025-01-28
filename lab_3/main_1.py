import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 1) Загрузка данных
data = pd.read_csv('lab_3/abalone.csv')

# 2) Преобразование признака Sex в числовой
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

# 3) Разделение на признаки и целевую переменную
X = data.iloc[:, :-1].values  # все столбцы, кроме последнего
y = data.iloc[:, -1].values  # последний столбец

# 4) Обучение случайного леса с различным числом деревьев и оценка качества
min_trees = None
scores = []

for n_trees in range(1, 51):
    rfr = RandomForestRegressor(n_estimators=n_trees, random_state=1)

    # Кросс-валидация
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        rfr.fit(X_train, y_train)
        predictions = rfr.predict(X_test)
        r2 = r2_score(y_test, predictions)
        r2_scores.append(r2)

    # Запись среднего R2
    mean_r2 = np.mean(r2_scores)
    scores.append(mean_r2)

    if min_trees is None and mean_r2 > 0.52:
        min_trees = n_trees

# 5) Определение минимального количества деревьев с R2 выше 0.52
print(f"Минимальное количество деревьев для R2 > 0.52: {min_trees}")

# 6) Визуализация изменения качества
plt.plot(range(1, 51), scores, marker='o')
plt.title('Изменение R2 в зависимости от количества деревьев')
plt.xlabel('Количество деревьев')
plt.ylabel('R2')
plt.grid()
plt.axhline(y=0.52, color='r', linestyle='--', label='R2 = 0.52')
plt.legend()
plt.show()