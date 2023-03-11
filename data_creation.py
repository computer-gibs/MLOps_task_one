import pandas as pd
import numpy as np
import os

# Создаем папки train и test, если они еще не существуют
if not os.path.exists('train'):
    os.mkdir('train')
if not os.path.exists('test'):
    os.mkdir('test')

# Наборы данных, которые мы хотим создать
datasets = [
    {'name': 'dataset1', 'length': 365, 'anomalies': True, 'noise': False},
    {'name': 'dataset2', 'length': 365, 'anomalies': False, 'noise': True},
    {'name': 'dataset3', 'length': 365, 'anomalies': True, 'noise': True}
]

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