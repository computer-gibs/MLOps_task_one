import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
import os

# Загружаем модель машинного обучения из файла
filename = 'models/model.pkl'
with open(filename, 'rb') as file:
    model = pickle.load(file)

# Проверяем точность модели на данных из папки "test"
test_data = []
for file in os.listdir('test'):
    if file.endswith('.csv'):
        data = pd.read_csv(f'test/{file}')
        X = data[['infected']]
        y = data['recovered']
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        test_data.append((file, mse))
print('Test data:')
for data in test_data:
    print(f'{data[0]} - MSE: {data[1]}')