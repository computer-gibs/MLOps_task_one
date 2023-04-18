import pandas as pd
from catboost.datasets import titanic
train_df, test_df = titanic()
train_df.to_csv('titanic.csv', index=False)
data = train_df[['Pclass', 'Sex', 'Age']].copy()
data.to_csv('titanic_age_sex.csv', index=False)
mean_age = data['Age'].mean()
data['Age'] = data['Age'].fillna(mean_age)
data.to_csv('titanic_age_sex_filled.csv', index=False)
one_hot = pd.get_dummies(data['Sex'], prefix='Sex')
data = pd.concat([data, one_hot], axis=1)
data.drop('Sex', axis=1, inplace=True)
data.to_csv('titanic_age_sex_filled_encoded.csv', index=False)
