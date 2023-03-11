import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

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