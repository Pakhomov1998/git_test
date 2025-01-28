import pandas as pd

# Загрузка данных
data_train = pd.read_csv('lab_1\salary-train.csv')
data_test = pd.read_csv('lab_1\salary-test-mini.csv')



import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack

# a) Приведение текстов к нижнему регистру и очистка текста
def preprocess_text(text):
    return re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

data_train['Description'] = data_train['FullDescription'].apply(preprocess_text)
data_test['Description'] = data_test['FullDescription'].apply(preprocess_text)

# b) Замена пропусках в столбцах LocationNormalized и ContractTime
data_train['LocationNormalized'].fillna('nan', inplace=True)
data_train['ContractTime'].fillna('nan', inplace=True)
data_test['LocationNormalized'].fillna('nan', inplace=True)
data_test['ContractTime'].fillna('nan', inplace=True)

# c) Применение TfidfVectorizer для текстов
tfidf_vectorizer = TfidfVectorizer(min_df=5)
X_train_text = tfidf_vectorizer.fit_transform(data_train['Description'])
X_test_text = tfidf_vectorizer.transform(data_test['Description'])

# d) Применение DictVectorizer для категориальных признаков
enc = DictVectorizer()
X_train_categ = enc.fit_transform(data_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test_categ = enc.transform(data_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

# e) Объединение всех признаков
X_train = hstack([X_train_text, X_train_categ])
X_test = hstack([X_test_text, X_test_categ])

from sklearn.linear_model import Ridge

# Целевая переменная
y_train = data_train['SalaryNormalized']

# Обучение модели
ridge_model = Ridge(alpha=1)
ridge_model.fit(X_train, y_train)

# Прогнозы для тестовых данных
predictions = ridge_model.predict(X_test)

# Вывод прогнозов через пробел
print(' '.join(map(str, predictions)))