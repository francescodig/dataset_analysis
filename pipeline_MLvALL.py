import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.decomposition import PCA

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

def run_pipeline(train_path, test_path):
    print(f"\n==================================================")
    print(f"AVVIO PIPELINE PER: {train_path.stem}")
    print(f"==================================================")

    # 1. Caricamento Dati
    df_train = load_data(train_path)
    df_test  = load_data(test_path)

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

    # 7. Label Encoding del Target
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_text)
    y_test = le.transform(y_test_text)

    # 8. Analisi delle Componenti Principali (PCA)
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)

    # Decommentare per mostrare i grafici ad ogni iterazione
    # plt.figure(figsize=(8, 6))
    # plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='viridis', alpha=0.5)
    # plt.title(f'PCA - {train_path.stem}')
    # plt.show()

    # 9. Apprendimento Non Supervisionato: K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_train_scaled)
    sil_score = silhouette_score(X_train_scaled, cluster_labels)
    print(f"Silhouette Score (K=3): {sil_score:.4f}\n")

    # 10. Apprendimento Supervisionato
    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_scaled, y_train)
    dt_preds = dt.predict(X_test_scaled)
    print("--- Decision Tree Performance ---")
    print(classification_report(y_test, dt_preds, labels=range(len(le.classes_)), target_names=le.classes_, zero_division=0))

    # Neural Network (MLP)
    mlp = MLPClassifier(random_state=42, max_iter=500)
    mlp.fit(X_train_scaled, y_train)
    mlp_preds = mlp.predict(X_test_scaled)
    print("--- Neural Network (MLP) Performance ---")
    print(classification_report(y_test, mlp_preds, labels=range(len(le.classes_)), target_names=le.classes_, zero_division=0))

# --- ESECUZIONE BATCH SU TUTTA LA DIRECTORY ---

# Sostituisci '.' con il percorso effettivo della tua cartella se necessario
data_dir = Path('.') 

# Cerca dinamicamente tutti i file che terminano in .data
for train_file in data_dir.glob("*.data"):
    # Costruisci il nome del file di test previsto (.data -> .test)
    test_file = train_file.with_suffix('.test')
    
    if test_file.exists():
        try:
            # Tenta di eseguire la pipeline per la coppia trovata
            run_pipeline(train_file, test_file)
        except Exception as e:
            # Isola l'errore senza far crollare l'intero programma
            print(f"\n[ERRORE] Impossibile processare la coppia {train_file.name} / {test_file.name}")
            print(f"Motivo: {e}")
    else:
        print(f"\n[ATTENZIONE] File di test mancante per {train_file.name}. Ignorato.")