import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Наборы данных, которые мы хотим создать
datasets = [
    {'name': 'dataset1', 'length': 365, 'anomalies': True, 'noise': False},
    {'name': 'dataset2', 'length': 365, 'anomalies': False, 'noise': True},
    {'name': 'dataset3', 'length': 365, 'anomalies': True, 'noise': True}
]

# Создаем папки train и test, если они еще не существуют
if not os.path.exists('train'):
    os.mkdir('train')
if not os.path.exists('test'):
    os.mkdir('test')

# Создаем каждый набор данных
for dataset in datasets:
    # Создаем DataFrame с данными
    data = pd.DataFrame({'date': pd.date_range(start='2020-01-01', periods=dataset['length'], freq='D')})
    data['infected'] = np.random.randint(100, 1000, size=dataset['length'])
    data['recovered'] = np.random.randint(50, 500, size=dataset['length'])

    # Добавляем аномалии
    if dataset['anomalies']:
        data.loc[100:110, 'infected'] = np.random.randint(2000, 5000, size=11)
        data.loc[200:210, 'recovered'] = np.random.randint(1000, 2000, size=11)

    # Добавляем шум
    if dataset['noise']:
        data['infected'] = data['infected'].apply(lambda x: x + np.random.randint(-100, 100))
        data['recovered'] = data['recovered'].apply(lambda x: x + np.random.randint(-50, 50))

    # Сохраняем данные в csv-файл
    if dataset['name'] in ['dataset1', 'dataset2']:
        data.to_csv(f'train/{dataset["name"]}.csv', index=False)
    else:
        data.to_csv(f'test/{dataset["name"]}.csv', index=False)


# Создаем папки processed_train и processed_test, если они еще не существуют
if not os.path.exists('processed_train'):
    os.mkdir('processed_train')
if not os.path.exists('processed_test'):
    os.mkdir('processed_test')

# Наборы данных, которые мы хотим обработать
datasets = [
    {'name': 'dataset1', 'dir': 'train', 'processed_dir': 'processed_train'},
    {'name': 'dataset2', 'dir': 'train', 'processed_dir': 'processed_train'},
    {'name': 'dataset3', 'dir': 'test', 'processed_dir': 'processed_test'}
]

# Обрабатываем каждый набор данных
for dataset in datasets:
    # Загружаем данные из csv-файла
    data = pd.read_csv(f'{dataset["dir"]}/{dataset["name"]}.csv')

    # Выполняем масштабирование данных
    scaler = StandardScaler()
    data['infected'] = scaler.fit_transform(data['infected'].values.reshape(-1, 1))
    data['recovered'] = scaler.fit_transform(data['recovered'].values.reshape(-1, 1))

    # Сохраняем обработанные данные в новый csv-файл
    data.to_csv(f'{dataset["processed_dir"]}/{dataset["name"]}_processed.csv', index=False)


# Загружаем данные из csv-файлов в DataFrame
data1 = pd.read_csv('processed_train/dataset1_processed.csv')
data2 = pd.read_csv('processed_train/dataset2_processed.csv')

# Объединяем данные в один DataFrame
data = pd.concat([data1, data2], ignore_index=True)

# Выделяем признаки и целевые переменные
X = data[['infected']]
y = data['recovered']

# Создаем и обучаем модель машинного обучения
model = LinearRegression()
model.fit(X, y)

# Сохраняем модель в файл
if not os.path.exists('models'):
    os.mkdir('models')
filename = 'models/model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)


# Проверяем точность модели на данных из папки "train"
train_data = []
for file in os.listdir('processed_train'):
    if file.endswith('_processed.csv'):
        data = pd.read_csv(f'processed_train/{file}')
        X = data[['infected']]
        y = data['recovered']
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        train_data.append((file, mse))
print('Train data:')
for data in train_data:
    print(f'{data[0]} - MSE: {data[1]}')


# Проверяем точность модели на данных из папки "test"
test_data = []
for file in os.listdir('processed_test'):
    if file.endswith('_processed.csv'):
        data = pd.read_csv(f'processed_test/{file}')
        X = data[['infected']]
        y = data['recovered']
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        test_data.append((file, mse))
print('Test data:')
for data in test_data:
    print(f'{data[0]} - MSE: {data[1]}')