import pandas as pd

HITS_PATH     = r"C:\Users\Никита\Skillfactory\Project workshop (hackathon)\ga_hits.pkl"
SESSIONS_PATH = r"C:\Users\Никита\Skillfactory\Project workshop (hackathon)\ga_sessions.csv"
OUTPUT_PATH   = r"C:\Users\Никита\Skillfactory\Project workshop (hackathon)\df_target.csv"

# 1. Загрузка данных
sessions = pd.read_csv(SESSIONS_PATH, low_memory=False)
hits = pd.read_pickle(HITS_PATH)
print(f"hits:     {hits.shape}")
print(f"sessions: {sessions.shape}")

# 2. Определяем целевые действия
target_events = [
    ('sub_submit',        'sub_submit_success'),
    ('sub_button_click',  'sub_car_claim_submit_click'),
    ('sub_button_click',  'sub_callback_submit_click'),
    ('sub_button_click',  'sub_car_request_submit_click'),
    ('greenday_sub_submit', 'greenday_sub_submit_success'),
    ('phone',             'form_request_call_sent'),
]

# Создаём маску для целевых событий
mask = pd.Series(False, index=hits.index)
for cat, action in target_events:
    mask |= (hits['event_category'] == cat) & (hits['event_action'] == action)
print(f"\nВсего целевых событий: {mask.sum()}")

# 3. Создаём целевую переменную по SESSION_ID
#Для каждой сессии: 1 если было хотя бы одно целевое действие, иначе 0
target = (
    hits[mask]
    .groupby('session_id')
    .size()
    .reset_index(name='target_count')
    .assign(target=1)
    [['session_id', 'target']]
)
print(f"Сессий с целевым действием: {len(target)}")

# 4 Объединяем с сессиями
df = sessions.merge(target, on='session_id', how='left')
df['target'] = df['target'].fillna(0).astype(int)

print(f"\nИтоговый датасет: {df.shape}")
print(f"Целевых сессий (target=1): {df['target'].sum()} ({df['target'].mean()*100:.2f}%)")
print(f"Не целевых (target=0):     {(df['target']==0).sum()}")


# 5. Сохраняем
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nФайл сохранён: {OUTPUT_PATH}")
print("\nПервые 3 строки:")
print(df[['session_id', 'visit_date', 'utm_source', 'device_category', 'target']].head(3))
