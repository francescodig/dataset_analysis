import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.decomposition import PCA

# Necessario per la SMOTE (pip install imbalanced-learn)
from imblearn.over_sampling import SMOTE

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

# 5. Imputazione dei Valori Mancanti
imputer = SimpleImputer(strategy='median')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# 6. Scaling (Standardizzazione)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train_imputed.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test_imputed.columns)

# 7. Label Encoding del Target per i modelli supervisionati
le = LabelEncoder()
y_train = le.fit_transform(y_train_text)
y_test = le.transform(y_test_text)

# 8. Analisi delle Componenti Principali (PCA)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', alpha=0.5)
plt.xlabel('Prima Componente Principale')
plt.ylabel('Seconda Componente Principale')
plt.title('Proiezione PCA del dataset Tiroideo')
plt.show()

# 9. Apprendimento Non Supervisionato: K-Means
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_train_scaled)

# Valutazione tramite Silhouette Score
sil_score = silhouette_score(X_train_scaled, cluster_labels)
print(f"Silhouette Score (K=3): {sil_score:.4f}")

# Integrazione SMOTE per il bilanciamento del Training Set
smote = SMOTE(random_state=42, k_neighbors=1)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# 10. Apprendimento Supervisionato su Dati Bilanciati
# A. Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train_smote, y_train_smote)
dt_preds = dt.predict(X_test_scaled)

print("--- Decision Tree Performance (con SMOTE) ---")
print(f"Accuracy: {accuracy_score(y_test, dt_preds):.4f}")
print(classification_report(y_test, dt_preds, labels=range(len(le.classes_)), target_names=le.classes_, zero_division=0))

# B. Multi-Layer Perceptron (Rete Neurale)
mlp = MLPClassifier(random_state=42, max_iter=500)
mlp.fit(X_train_smote, y_train_smote)
mlp_preds = mlp.predict(X_test_scaled)

print("--- Neural Network (MLP) Performance (con SMOTE) ---")
print(f"Accuracy: {accuracy_score(y_test, mlp_preds):.4f}")
print(classification_report(y_test, mlp_preds, labels=range(len(le.classes_)), target_names=le.classes_, zero_division=0))