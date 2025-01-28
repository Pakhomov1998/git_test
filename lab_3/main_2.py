import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

# 1. Загрузка данных и разбиение на обучающую и тестовую выборки
data = pd.read_csv('lab_3/gbm-data.csv')
X = data.iloc[:, 1:].values  # Все колонки, кроме первой
y = data.iloc[:, 0].values    # Первая колонка - целевая переменная

# Разбиение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

# 2. Обучение GradientBoostingClassifier и вычисление log-loss
learning_rates = [1, 0.5, 0.3, 0.2, 0.1]
best_log_loss = float('inf')
best_iteration = -1
best_learning_rate = None

for learning_rate in learning_rates:
    gbc = GradientBoostingClassifier(n_estimators=250, learning_rate=learning_rate, verbose=True, random_state=241)
    gbc.fit(X_train, y_train)

    train_loss = []
    test_loss = []

    # Вычисление log-loss на обучающей выборке
    for y_pred in gbc.staged_decision_function(X_train):
        prob = 1 / (1 + np.exp(-y_pred))  # Преобразование в вероятности
        train_loss.append(log_loss(y_train, prob))

    # Вычисление log-loss на тестовой выборке
    for y_pred in gbc.staged_decision_function(X_test):
        prob = 1 / (1 + np.exp(-y_pred))  # Преобразование в вероятности
        test_loss.append(log_loss(y_test, prob))

    # Поиск минимального значения log-loss на тестовой выборке
    min_test_loss = min(test_loss)
    min_test_iteration = test_loss.index(min_test_loss)

    # Обновление наилучших значений
    if min_test_loss < best_log_loss:
        best_log_loss = min_test_loss
        best_iteration = min_test_iteration
        best_learning_rate = learning_rate

    # Построение графика
    plt.figure()
    plt.plot(train_loss, 'g', linewidth=2, label='train')
    plt.plot(test_loss, 'r', linewidth=2, label='test')
    plt.legend()
    plt.title(f'Learning Rate: {learning_rate}')
    plt.xlabel('Iterations')
    plt.ylabel('Log-loss')
    plt.show()

# 3. Вывод результатов
print(f'Best log-loss: {best_log_loss} at iteration: {best_iteration} with learning_rate: {best_learning_rate}')

# 4. Минимальное значение log-loss и номер итерации для learning_rate = 0.2
learning_rate = 0.2
gbc = GradientBoostingClassifier(n_estimators=250, learning_rate=learning_rate, verbose=True, random_state=241)
gbc.fit(X_train, y_train)

train_loss = []
test_loss = []

for y_pred in gbc.staged_decision_function(X_train):
    prob = 1 / (1 + np.exp(-y_pred))
    train_loss.append(log_loss(y_train, prob))

for y_pred in gbc.staged_decision_function(X_test):
    prob = 1 / (1 + np.exp(-y_pred))
    test_loss.append(log_loss(y_test, prob))

min_test_loss_02 = min(test_loss)
min_test_iteration_02 = test_loss.index(min_test_loss_02)

print(f'Min log-loss for learning_rate=0.2: {min_test_loss_02} at iteration: {min_test_iteration_02}')

# 5. Обучение RandomForestClassifier и оценка log-loss
n_estimators_rf = best_iteration + 1  # +1, так как индексация начинается с 0
rfc = RandomForestClassifier(n_estimators=n_estimators_rf, random_state=241)
rfc.fit(X_train, y_train)

# Предсказание вероятностей
pred_proba = rfc.predict_proba(X_test)
log_loss_rf = log_loss(y_test, pred_proba)

print(f'Log-loss for RandomForestClassifier: {log_loss_rf}')