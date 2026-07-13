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
from sklearn.metrics import accuracy_score, classification_report, silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

column_names = [
    'age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
    'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
    'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH_measured', 'TSH', 'T3_measured', 'T3', 'TT4_measured', 'TT4',
    'T4U_measured', 'T4U', 'FTI_measured', 'FTI', 'TBG_measured', 'TBG',
    'referral_source', 'target'
]

# header=None perchè la prima riga del dataset non è l'intestazione
# names=column_names perchè assegnamo alle colonne i nomi che abbiamo estratto precedentemente
# dato che il target è solitamente negative.|3454 oppure hypothyroid.|1289, 
# estraiamo solamente l'esito finale. 
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

# 5. Label Encoding 
le = LabelEncoder()
y_train = le.fit_transform(y_train_text)
y_test  = le.transform(y_test_text)

# 6. Imputazione dei dati mancanti
imp = SimpleImputer(strategy='median')
X_train_imp = pd.DataFrame(imp.fit_transform(X_train), columns=X_train.columns)
X_test_imp  = pd.DataFrame(imp.transform(X_test), columns=X_test.columns)

# 7. Rimozione Outlier
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

# --- NUOVA SEZIONE: Analisi Esplorativa ---
print("--- Generazione Grafici Analisi Esplorativa ---")
plt.figure(figsize=(10, 8))
sns.heatmap(X_train_clean[num_cols].corr(), annot=True, cmap='viridis', fmt='.2f')
plt.title('Matrice di Correlazione (Variabili Numeriche)')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=X_train_clean[num_cols])
plt.title('Distribuzione Variabili Numeriche (Post-Pulizia)')
plt.show()

# 8. Scalatura dei dati
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_clean)
X_test_sc  = scaler.transform(X_test_imp)

# --- NUOVA SEZIONE: Riduzione della Dimensionalità (PCA) ---
print("--- Generazione Bi-Plot PCA ---")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_sc)

plt.figure(figsize=(8, 6))
# Il parametro c=y_train_clean permette di visualizzare i target nello spazio PCA
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, c=y_train_clean, cmap='coolwarm', s=10)
plt.title('Proiezione PCA (Bi-Plot)')
plt.xlabel('Componente Principale 1')
plt.ylabel('Componente Principale 2')
plt.colorbar(label='Target Class')
plt.show()

# 9. Addestramento e Valutazione Modelli
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=42),
}

print("--- Report Dettagliato dei Modelli ---")
for name, model in models.items():
    model.fit(X_train_sc, y_train_clean)
    pred = model.predict(X_test_sc)
    print(f"\nModello: {name}")
    
    print(classification_report(
        y_test, 
        pred, 
        labels=range(len(le.classes_)), 
        target_names=le.classes_,
        zero_division=0
    ))

# 10. Clustering con KMeans
# 10. Clustering con KMeans
K = 4
km = KMeans(n_clusters=K, random_state=42, n_init=10)
cluster_labels = km.fit_predict(X_train_sc)
sil_score = silhouette_score(X_train_sc, cluster_labels)

print(f"\nKMeans completato. Cluster unici trovati: {np.unique(cluster_labels)}")
print(f"Silhouette Score (K={K}): {sil_score:.4f}")

# --- NUOVA SEZIONE: Visualizzazione Silhouette ---
print("--- Generazione Grafico Silhouette ---")
sample_silhouette_values = silhouette_samples(X_train_sc, cluster_labels)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
y_lower = 10 

for i in range(K):
    # Aggrega e ordina le silhouette del cluster i-esimo
    cluster_silhouette_vals = sample_silhouette_values[cluster_labels == i]
    cluster_silhouette_vals.sort()
    
    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = cm.nipy_spectral(float(i) / K)
    # fill_betweenx è molto più efficiente di barh per dataset di grandi dimensioni
    ax[0].fill_betweenx(np.arange(y_lower, y_upper),
                        0, cluster_silhouette_vals,
                        facecolor=color, edgecolor=color, alpha=0.7)
    
    # Etichetta del cluster al centro del blocco
    ax[0].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
    # Spazio per il cluster successivo
    y_lower = y_upper + 10

ax[0].set_title("Profilo Silhouette per i Cluster")
ax[0].set_xlabel("Valori del coefficiente di Silhouette")
ax[0].set_ylabel("Etichetta del Cluster")

# La linea verde rappresenta lo score medio
ax[0].axvline(x=sil_score, color="green", linestyle="--", linewidth=2)

# Scatter plot usando le coordinate PCA calcolate nella Sezione 8
colors = cm.nipy_spectral(cluster_labels.astype(float) / K)
ax[1].scatter(X_pca[:, 0], X_pca[:, 1], marker='.', s=30, alpha=0.7, c=colors, edgecolor='k')

ax[1].set_title("Proiezione dei Cluster (Spazio PCA)")
ax[1].set_xlabel("Componente Principale 1")
ax[1].set_ylabel("Componente Principale 2")

plt.suptitle(f"Analisi K-Means (K={K})", fontweight='bold')
plt.tight_layout()
plt.show()