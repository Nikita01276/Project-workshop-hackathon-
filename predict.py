import pickle
import pandas as pd
import numpy as np

# Загружаем модель
MODEL_PATH = r'C:\Users\Никита\Skillfactory\Project workshop (hackathon)\best_model.pkl'
DATA_PATH  = r'C:\Users\Никита\Skillfactory\Project workshop (hackathon)\df_clean.csv'

with open(MODEL_PATH, 'rb') as f:
    bundle = pickle.load(f)

model    = bundle['model']
features = bundle['features']

# Загружаем данные для target encoding (средние конверсии по каналам/городам)
df = pd.read_csv(DATA_PATH, low_memory=False)
medium_conv = df.groupby('utm_medium')['target'].mean().to_dict()
source_conv = df.groupby('utm_source')['target'].mean().to_dict()
city_conv   = df.groupby('geo_city')['target'].mean().to_dict()
global_conv = df['target'].mean()  # fallback для неизвестных значений

def preprocess(session: dict) -> pd.DataFrame:
    """
    Принимает словарь с данными визита,
    возвращает DataFrame с признаками для модели.
    """
    utm_medium = session.get('utm_medium', 'unknown')
    utm_source = session.get('utm_source', 'unknown')
    geo_city   = session.get('geo_city', 'unknown')
    visit_number = float(session.get('visit_number', 1))
    visit_time   = float(session.get('visit_time', 0))
    visit_date   = pd.to_datetime(session.get('visit_date', '2021-01-01'))
    device       = session.get('device_category', 'mobile')

    hour = int(visit_time // 3600) % 24

    row = {
        'visit_number':      visit_number,
        'visit_number_log':  np.log1p(visit_number),
        'month':             visit_date.month,
        'day_of_week':       visit_date.dayofweek,
        'hour':              hour,
        'is_first_visit':    int(visit_number == 1),
        'is_repeat_visit':   int(visit_number > 1),
        'is_desktop':        int(device == 'desktop'),
        'is_mobile':         int(device == 'mobile'),
        'is_cpa':            int(utm_medium == 'cpa'),
        'is_smm':            int(utm_medium == 'smm'),
        'is_referral':       int(utm_medium == 'referral'),
        'is_organic':        int(utm_medium == 'organic'),
        'is_cpc':            int(utm_medium == 'cpc'),
        'is_push':           int(utm_medium == 'push'),
        'is_weekend':        int(visit_date.dayofweek >= 5),
        'is_work_hours':     int(9 <= hour <= 18),
        'is_moscow':         int('moscow' in geo_city.lower() or 'moskva' in geo_city.lower()),
        'utm_medium_conv':   medium_conv.get(utm_medium, global_conv),
        'utm_source_conv':   source_conv.get(utm_source, global_conv),
        'city_conv':         city_conv.get(geo_city, global_conv),
    }

    return pd.DataFrame([row])[features]


def predict(session: dict) -> dict:
    """
    Принимает словарь с данными визита.
    Возвращает: prediction (0 или 1) и probability.
    """
    X = preprocess(session)
    proba = model.predict_proba(X)[0][1]
    pred  = int(proba >= 0.5)
    return {
        'prediction':  pred,
        'probability': round(proba, 4),
        'message':     'Целевое действие ожидается' if pred == 1 else 'Целевое действие не ожидается'
    }


# Пример использования
if __name__ == '__main__':
    # Тест 1 — тёплый пользователь через smm
    session_1 = {
        'utm_source':      'some_source',
        'utm_medium':      'smm',
        'geo_city':        'Moscow',
        'device_category': 'desktop',
        'visit_number':    4,
        'visit_time':      43200,
        'visit_date':      '2021-11-15',
    }

    # Тест 2 — холодный пользователь через cpc
    session_2 = {
        'utm_source':      'unknown_source',
        'utm_medium':      'cpc',
        'geo_city':        'Saratov',
        'device_category': 'mobile',
        'visit_number':    1,
        'visit_time':      3600,
        'visit_date':      '2021-08-10',
    }

    print('Тест 1 (тёплый пользователь, smm, desktop, 4-й визит):')
    print(predict(session_1))

    print('\nТест 2 (холодный пользователь, cpc, mobile, 1-й визит):')
    print(predict(session_2))
