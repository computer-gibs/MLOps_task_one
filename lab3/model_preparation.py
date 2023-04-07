import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

# Загружаем данные из csv-файлов в DataFrame
data1 = pd.read_csv('train/dataset1.csv')
data2 = pd.read_csv('train/dataset2.csv')

# Объединяем данные в один DataFrame
data = pd.concat([data1, data2], ignore_index=True)

# Выделяем признаки и целевые переменные
X = data[['infected']]
y = data['recovered']

# Создаем и обучаем модель машинного обучения
model = LinearRegression()
model.fit(X, y)

# Проверяем точность модели на данных из папки "train"
train_data = []
for file in os.listdir('train'):
    if file.endswith('.csv'):
        data = pd.read_csv(f'train/{file}')
        X = data[['infected']]
        y = data['recovered']
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        train_data.append((file, mse))
print('Train data:')
for data in train_data:
    print(f'{data[0]} - MSE: {data[1]}')

# Сохраняем модель в файл
if not os.path.exists('models'):
    os.mkdir('models')
filename = 'models/model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)