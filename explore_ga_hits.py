import pandas as pd

PATH = r"C:\Users\Никита\Skillfactory\Project workshop (hackathon)\ga_hits.pkl" 

df = pd.read_csv(PATH, low_memory=False)

    
    df = pd.read_pickle(PATH)
print(f"Размер датасета: {df.shape}")

#1. Все уникальные комбинации event_category + event_action
print("\n1 Уникальные комбинации event_category + event_action:")
combo = df.groupby(['event_category', 'event_action']).size().reset_index(name='count')
combo = combo.sort_values('count', ascending=False)
print(combo.to_string(index=False))

# 2 Уникальные event_label
print("\n2 Уникальные event_label (топ-50 по частоте:")
print(df['event_label'].value_counts().head(50))

# 3. Уникальные hit_page_path
print("\n3 Страницы с ключевыми словами (form, lead, success, spasibo, zayavka, thanks):")
keywords = ['form', 'lead', 'success', 'spasibo', 'zayavka', 'thanks', 'order', 'submit']
mask = df['hit_page_path'].str.lower().str.contains('|'.join(keywords), na=False)
print(df.loc[mask, 'hit_page_path'].value_counts().head(20))

# 4. Находим целевые события
print("\n4 Поиск по event_action — 'submit', 'send', 'lead', 'form', 'order', 'call':")
target_keywords = ['submit', 'send', 'lead', 'form', 'order', 'call', 'click', 'complete', 'success', 'zayavka']
mask2 = df['event_action'].str.lower().str.contains('|'.join(target_keywords), na=False)
print(df.loc[mask2, ['event_category', 'event_action', 'event_label']].value_counts().head(30))

