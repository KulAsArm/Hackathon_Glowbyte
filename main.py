import os
import re

# Математические библиотеки
import pandas as pd

# Библиотека для загрузки модели
from joblib import load

# глобальные настройки
CATEGORY_FEATURES = ['rain_snow', 'wing', 'summer', 'cloudy', 'date', 'dayofweek', 'month', 'year', 'hour']


def index_date(data):
    """
    Преобразование даты
    """
    data['time'] = pd.to_datetime(data['time'], format='%H').dt.time
    data['date_time'] = pd.to_datetime(
        data['date'] + 'T' + data['time'].astype(str),
        format='%Y-%m-%dT%H:%M:%S'
    )
    data = data.set_index('date_time')
    # print(['Хронологический порядок индекса отсутствует', 'Индекс соответствует хронологическому порядку'][data.index.is_monotonic])
    data.drop(columns=['date', 'time'], axis=1, inplace=True)
    return data


def rainsnow(x):
    """
    Вероятность осадков
    """
    if len(re.findall(r'\b\d+\b', x)) != 0:
        temp = re.findall(r'\b\d+\b', x)
        result = int(temp[0])
    elif 'дожд' in x or 'снег' in x or 'ливень' in x or 'снегопад' in x:
        result = 100
    else:
        result = 0
    return result


def make_features(data, max_lag, rolling_mean_size):
    """
    Создание признаков
    """
    data['date'] = data.index.day
    data['dayofweek'] = data.index.dayofweek
    data['month'] = data.index.month
    data['year'] = data.index.year
    data['hour'] = data.index.hour

    for lag in range(1, max_lag + 1):
        data['lag_{}'.format(lag)] = data['target'].shift(lag)

    data['rolling_mean'] = data['target'].shift().rolling(rolling_mean_size).mean()


def preprocess(path: str):
    df = pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df = index_date(df)
        df = df.ffill(axis=0)
        df.isnull().sum()
        # вероятность осадков
        df['rain_snow'] = df['weather_pred'].apply(lambda x: rainsnow(x)) / 100
        # ветер
        df['wing'] = df['weather_pred'].apply(lambda x: [0, 1]['ветер' in x])
        # ясно/солнечно
        df['summer'] = df['weather_pred'].apply(lambda x: [0, 1]['ясно' in x or 'солнечно' in x])
        # пасмурно
        df['cloudy'] = df['weather_pred'].apply(lambda x: [0, 1]['пас' in x])
        # удаление неиформативных столбцов
        df = df.drop(columns=['temp', 'weather_pred', 'weather_fact'], axis=1)
    except ValueError:
        print('Выбран неверный файл!')
        exit()

    df_proba = df.copy()
    make_features(df_proba, 10, 10)

    X_train = df_proba.drop('target', axis=1)
    y_train = df_proba['target']

    X_train_cat = X_train.copy()

    for i in [X_train_cat]:
        for j in CATEGORY_FEATURES:
            i[j] = i[j].astype('category')

    return X_train_cat, y_train


def group_by_days(prediction: list):
    df_predict_lgb = pd.DataFrame(data=prediction, index=x.index, columns=['predict'])
    df_result = df_predict_lgb.resample('1D').sum()
    df_result.reset_index(inplace=True)
    df_result = df_result.rename(columns={"date_time": "date"})
    print(df_result.columns)
    df_result.to_csv('result.csv')


def input_path():
    path = input('Введите путь до приватного датасета: \n')
    if not os.path.exists(path):
        input_path()
    else:
        return path


if __name__ == '__main__':
    path = input_path()
    x = preprocess(path)[0]
    print('Пометка: Файл модели должен находится в одной папке с файлом main.py!')
    model = load('Dmitriy.joblib')
    predict_list = model.predict(x)
    group_by_days(predict_list)
