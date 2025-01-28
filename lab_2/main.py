import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# 1) Загрузка данных
close_prices = pd.read_csv('lab_2/close_prices.csv')
djia_index = pd.read_csv('lab_2/djia_index.csv')

# Проверка загруженных данных
print("Данные о ценах акций:")
print(close_prices.head())
print("\nДанные об индексе Доу-Джонса:")
print(djia_index.head())

# Удаление столбца с датами
close_prices = close_prices.drop(columns=['date'])

# 2) Обучение PCA с числом компонент равным 10
pca = PCA(n_components=10)
pca.fit(close_prices)

# Сколько компонент хватит, чтобы объяснить 90% дисперсии?
explained_variance = np.cumsum(pca.explained_variance_ratio_)
n_components_90 = np.argmax(explained_variance >= 0.90) + 1
print(f"Количество компонент, необходимых для объяснения 90% дисперсии: {n_components_90}")

# 3) Применение PCA к исходным данным и получение значений первой компоненты
first_component = pca.transform(close_prices)[:, 0]
print("Значения первой компоненты:")
print(first_component)

# 4) Корреляция Пирсона между первой компонентой и индексом Доу-Джонса
correlation = np.corrcoef(first_component, djia_index['^DJI'])[0, 1]
print(f"Корреляция Пирсона между первой компонентой и индексом Доу-Джонса: {correlation:.2f}")

# 5) Определение компании с наибольшим весом в первой компоненте
weights = pca.components_[0]
max_weight_index = np.argmax(weights)
companies = close_prices.columns
max_weight_company = companies[max_weight_index]
max_weight_value = weights[max_weight_index]

print(f"Компания с наибольшим весом в первой компоненте: {max_weight_company}, вес: {max_weight_value:.2f}")