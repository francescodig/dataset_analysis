import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report

column_names = [
    'age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
    'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
    'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH_measured', 'TSH', 'T3_measured', 'T3', 'TT4_measured', 'TT4',
    'T4U_measured', 'T4U', 'FTI_measured', 'FTI', 'TBG_measured', 'TBG',
    'referral_source', 'target'
]

def load_data(path):
    df = pd.read_csv(path, header=None, names=column_names, na_values='?')
    df['target'] = df['target'].str.split('|').str[0].str.strip()
    return df

# 1. Caricamento Dati
df_train = load_data('allhypo.data')
df_test  = load_data('allhypo.test')

# Rimozione colonna TBG interamente vuota
df_train = df_train.drop(columns=['TBG', 'TBG_measured'])
df_test  = df_test.drop(columns=['TBG', 'TBG_measured'])

# 2. Codifica Variabili Binarie
binary_cols = ['on_thyroxine','query_on_thyroxine','on_antithyroid_medication',
               'sick','pregnant','thyroid_surgery','I131_treatment','query_hypothyroid',
               'query_hyperthyroid','lithium','goitre','tumor','hypopituitary','psych',
               'TSH_measured','T3_measured','TT4_measured','T4U_measured','FTI_measured']

for c in binary_cols:
    df_train[c] = (df_train[c] == 't').astype(int)
    df_test[c]  = (df_test[c]  == 't').astype(int)

# 3. One-Hot Encoding
df_train = pd.get_dummies(df_train, columns=['sex', 'referral_source'], drop_first=False)
df_test  = pd.get_dummies(df_test,  columns=['sex', 'referral_source'], drop_first=False)
df_train, df_test = df_train.align(df_test, join='left', axis=1, fill_value=0)

# 4. Separazione Target e Features
X_train = df_train.drop(columns=['target'])
y_train_text = df_train['target']
X_test  = df_test.drop(columns=['target'])
y_test_text  = df_test['target']

# 5. Label Encoding (Novità introdotta)
le = LabelEncoder()
y_train = le.fit_transform(y_train_text)
y_test  = le.transform(y_test_text)

# 6. Imputazione dei dati mancanti (Spostata PRIMA della rimozione outlier)
imp = SimpleImputer(strategy='median')

# È fondamentale riconvertire l'output in DataFrame per mantenere i nomi delle colonne
X_train_imp = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns)
X_test_imp  = pd.DataFrame(imp.transform(X_test), columns=X_test.columns)

# 7. Rimozione Outlier (Ora sicura grazie all'imputazione precedente)
num_cols = ['age','TSH','T3','TT4','T4U','FTI']
mask = pd.Series(True, index=X_train_imp.index)

for c in num_cols:
    mean, std = X_train_imp[c].mean(), X_train_imp[c].std()
    col_mask = (X_train_imp[c] - mean).abs() <= 3 * std
    mask = mask & col_mask

X_train_clean = X_train_imp[mask]
y_train_clean = y_train[mask]

print(f"Dimensioni Training Set originarie: {X_train.shape}")
print(f"Dimensioni Training Set dopo pulizia outlier: {X_train_clean.shape}\n")

# 8. Scalatura dei dati
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_clean)
X_test_sc  = scaler.transform(X_test_imp)

# 9. Addestramento e Valutazione Modelli
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=42),
}

print("--- Risultati Accuratezza ---")
for name, model in models.items():
    model.fit(X_train_sc, y_train_clean)
    pred = model.predict(X_test_sc)
    print(f"{name}: {accuracy_score(y_test, pred):.4f}")

# 10. Clustering con KMeans
km = KMeans(n_clusters=4, random_state=42, n_init=10)
km.fit(X_train_sc)
print(f"\nKMeans completato. Cluster unici trovati: {np.unique(km.labels_)}")