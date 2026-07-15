import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# ===============================================================
# FASE 1: CARICAMENTO, DIZIONARIO ED ESPLORAZIONE
# ===============================================================
print("="*60)
print(" FASE 1: CARICAMENTO, DIZIONARIO ED ESPLORAZIONE")
print("="*60 + "\n")

column_names = [
    'age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
    'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
    'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH_measured', 'TSH', 'T3_measured', 'T3', 'TT4_measured', 'TT4',
    'T4U_measured', 'T4U', 'FTI_measured', 'FTI', 'TBG_measured', 'TBG',
    'referral_source', 'target'
]

# Definizione del dizionario delle feature
col_desc = {
    'age': 'Età del paziente (numerica)',
    'sex': 'Sesso del paziente (F/M)',
    'on_thyroxine': 'Assunzione di tiroxina (binaria)',
    'sick': 'Presenza di altre patologie (binaria)',
    'pregnant': 'Paziente in gravidanza (binaria)',
    'thyroid_surgery': 'Precedenti interventi alla tiroide (binaria)',
    'TSH': 'Ormone Tiroideo Stimolante (numerica - esame del sangue)',
    'T3': 'Triiodotironina (numerica - esame del sangue)',
    'TT4': 'Tiroxina totale (numerica - esame del sangue)',
    'target': 'Diagnosi clinica - Variabile da predire (stringa)'
}
print("Dizionario delle Feature Principali:")
print(pd.DataFrame.from_dict(col_desc, orient='index', columns=['Descrizione']))
print("\n")

def load_data(path):
    df = pd.read_csv(path, header=None, names=column_names, na_values='?')
    df['target'] = df['target'].str.split('|').str[0].str.strip()
    return df

# Caricamento
df_train = load_data('allhypo.data')
df_test  = load_data('allhypo.test')

# Rimozione colonne interamente vuote
df_train = df_train.drop(columns=['TBG', 'TBG_measured'])
df_test  = df_test.drop(columns=['TBG', 'TBG_measured'])

# Analisi dei valori mancanti
missing = pd.DataFrame({
    'train_missing': df_train.isnull().sum(),
    'test_missing' : df_test.isnull().sum(),
})
print('Colonne con valori mancanti prima dell\'imputazione:')
print(missing[missing.sum(axis=1) > 0])
print("\n")




# Unione temporanea per l'analisi del dataset globale iniziale
df_all = pd.concat([df_train, df_test], ignore_index=True)

print(f"Dimensioni totali del dataset (Train + Test): {df_all.shape[0]} pazienti, {df_all.shape[1]} feature.\n")

# Visualizzazione della distribuzione del dataset iniziale
sns.set_theme(style='whitegrid', palette='muted')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

target_counts = df_all['target'].value_counts()


# Grafico a Torta (Percentuali)
axes[0].pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set2"))
axes[0].set_title('Distribuzione Globale Target (Percentuali)', fontweight='bold')

# Grafico a Barre (Valori Assoluti)
sns.countplot(data=df_all, y='target', ax=axes[1], order=target_counts.index, palette="Set2", hue='target', legend=False)
axes[1].set_title('Distribuzione Globale Target (Conteggi Assoluti)', fontweight='bold')
axes[1].set_xlabel('Numero di Pazienti')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()

# ===============================================================
# 2. PREPROCESSING E MATRICE DI CORRELAZIONE
# ===============================================================
print("="*60)
print(" FASE 2: PREPROCESSING E MATRICE DI CORRELAZIONE")
print("="*60 + "\n")

# Mappatura variabili binarie
binary_cols = ['on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
               'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
               'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
               'TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured']

for c in binary_cols:
    df_train[c] = (df_train[c] == 't').astype(int)
    df_test[c]  = (df_test[c]  == 't').astype(int)

# One-Hot Encoding
df_train = pd.get_dummies(df_train, columns=['sex', 'referral_source'], drop_first=False)
df_test  = pd.get_dummies(df_test,  columns=['sex', 'referral_source'], drop_first=False)
df_train, df_test = df_train.align(df_test, join='left', axis=1, fill_value=0)

# Separazione X e y
X_train = df_train.drop(columns=['target'])
y_train_text = df_train['target']
X_test  = df_test.drop(columns=['target'])
y_test_text  = df_test['target']

# Imputazione
imputer = SimpleImputer(strategy='median')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Standardizzazione
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X_train_imputed.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test_imputed.columns)

# Label Encoding
le = LabelEncoder()
y_train = le.fit_transform(y_train_text)
y_test = le.transform(y_test_text)

print("Preprocessing completato con successo.\n")

print("Generazione della Matrice di Correlazione...\n")
plt.figure(figsize=(16, 12))

# Calcolo della matrice di correlazione
corr_matrix = X_train_scaled.corr()

# Creazione di una maschera per nascondere la metà superiore (simmetrica)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Heatmap
sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', vmin=-1, vmax=1, 
            annot=False, square=True, linewidths=.5)
plt.title('Matrice di Correlazione delle Feature (Pre-PCA)', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.show()


# ===============================================================
# 3. TASK A: CLUSTERING NON SUPERVISIONATO
# ===============================================================
print("="*60)
print(" FASE 3: CLUSTERING K-MEANS E OTTIMIZZAZIONE")
print("="*60 + "\n")

# Baseline
sil_scores = {}
for k in range(2, 7):
    km_k = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = km_k.fit_predict(X_train_scaled)
    sil_scores[k] = silhouette_score(X_train_scaled, labels_k)

# Feature Selection
rf_selector = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
rf_selector.fit(X_train_scaled, y_train)
importances = pd.Series(rf_selector.feature_importances_, index=X_train_scaled.columns)
importances_sorted = importances.sort_values(ascending=False)
top_features = importances_sorted.head(15).index.tolist()
X_train_selected = X_train_scaled[top_features]

# Visualizzazione: Feature Importances e Confronto PCA 2D
print("Generazione dei grafici per Feature Selection e PCA 2D...\n")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.barplot(x=importances_sorted.head(15).values, y=importances_sorted.head(15).index, ax=axes[0], hue=importances_sorted.head(15).index, palette='viridis', legend=False)
axes[0].set_title('Top 15 Feature Importanti', fontweight='bold')
axes[0].set_xlabel('Importanza (Random Forest)')

pca_2d = PCA(n_components=2, random_state=42)
X_pca_all = pca_2d.fit_transform(X_train_scaled)
scatter1 = axes[1].scatter(X_pca_all[:, 0], X_pca_all[:, 1], c=y_train, cmap='tab10', alpha=0.6)
axes[1].set_title('PCA 2D - Tutte le feature originali', fontweight='bold')
axes[1].set_xlabel('Componente Principale 1')
axes[1].set_ylabel('Componente Principale 2')

X_pca_sel = pca_2d.fit_transform(X_train_selected)
scatter2 = axes[2].scatter(X_pca_sel[:, 0], X_pca_sel[:, 1], c=y_train, cmap='tab10', alpha=0.6)
axes[2].set_title('PCA 2D - Solo feature selezionate', fontweight='bold')
axes[2].set_xlabel('Componente Principale 1')
axes[2].set_ylabel('Componente Principale 2')

plt.tight_layout()
plt.show()

# PCA (90% varianza) per il calcolo del clustering
pca_full = PCA(n_components=0.90, random_state=42)
X_train_pca_full = pca_full.fit_transform(X_train_selected)

sil_scores_pca = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = km.fit_predict(X_train_pca_full)
    sil_scores_pca[k] = silhouette_score(X_train_pca_full, labels_k)



# Plot Clustering
plt.figure(figsize=(10, 6))
plt.plot(list(sil_scores.keys()), list(sil_scores.values()), marker='s', linestyle='--', color='gray', label='1. Baseline (Tutte le feature)')
plt.plot(list(sil_scores_pca.keys()), list(sil_scores_pca.values()), marker='o', color='blue', label='2. FS + PCA (90%)')
plt.xlabel('Numero di cluster (k)')
plt.ylabel('Silhouette Score')
plt.title('Task A: Qualità del Clustering (Silhouette) per metodo di Resampling', fontweight='bold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ===============================================================
# 4. TASK B: CLASSIFICAZIONE SUPERVISIONATA
# ===============================================================
print("="*60)
print(" FASE 4: APPRENDIMENTO SUPERVISIONATO (TASK B)")
print("="*60 + "\n")

datasets = {}
datasets['Originale'] = (X_train_scaled, y_train)

rus_sup = RandomUnderSampler(random_state=42)
X_train_under_sup, y_train_under_sup = rus_sup.fit_resample(X_train_scaled, y_train)
datasets['Undersampling'] = (X_train_under_sup, y_train_under_sup)

ros_sup = RandomOverSampler(random_state=42)
X_train_ros_sup, y_train_ros_sup = ros_sup.fit_resample(X_train_scaled, y_train)
datasets['Random Oversampling'] = (X_train_ros_sup, y_train_ros_sup)

smote_sup = SMOTE(random_state=42, k_neighbors=1)
X_train_smote_sup, y_train_smote_sup = smote_sup.fit_resample(X_train_scaled, y_train)
datasets['SMOTE'] = (X_train_smote_sup, y_train_smote_sup)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Neural Network': MLPClassifier(random_state=42, max_iter=500)
}

results_f1 = {model_name: [] for model_name in models.keys()}
results_acc = {model_name: [] for model_name in models.keys()}
results_prec = {model_name: [] for model_name in models.keys()}
results_rec = {model_name: [] for model_name in models.keys()}

dataset_names = list(datasets.keys())

results_train_acc = {model_name: [] for model_name in models.keys()}
results_test_acc = {model_name: [] for model_name in models.keys()}
results_train_f1 = {model_name: [] for model_name in models.keys()}
results_test_f1 = {model_name: [] for model_name in models.keys()}

for data_name, (X_tr, y_tr) in datasets.items():
    print(f"--- Addestramento sul dataset: {data_name} ---")
    for model_name, model in models.items():
        model.fit(X_tr, y_tr)
        
        # Predizioni su Train e su Test
        y_pred_train = model.predict(X_tr)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calcolo Accuracy
        train_acc = accuracy_score(y_tr, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        # Calcolo F1-Macro
        train_f1 = f1_score(y_tr, y_pred_train, average='macro', zero_division=0)
        test_f1 = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
        
        results_train_acc[model_name].append(train_acc)
        results_test_acc[model_name].append(test_acc)
        results_train_f1[model_name].append(train_f1)
        results_test_f1[model_name].append(test_f1)
        
        print(f"{model_name:20s} -> Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f} (Train F1: {train_f1:.4f})")
    print()

# ===============================================================
# Generazione Grafici: Confronto Overfitting
# ===============================================================
x = np.arange(len(dataset_names))
width = 0.12  # Barre più sottili per raggruppare train e test per ogni modello

fig, axes = plt.subplots(2, 1, figsize=(14, 14))

# Plot 1: Train vs Test Accuracy
ax1 = axes[0]
multiplier = 0
for model_name in models.keys():
    offset = width * multiplier * 2.5
    
    # Train (con tratteggio per distinguerlo dal test)
    ax1.bar(x + offset, results_train_acc[model_name], width, label=f'{model_name} (Train)', alpha=0.6, hatch='//')
    # Test (tinta unita)
    ax1.bar(x + offset + width, results_test_acc[model_name], width, label=f'{model_name} (Test)')
    
    multiplier += 1

ax1.set_ylabel('Accuracy Score')
ax1.set_title('Confronto Overfitting: Train Accuracy vs Testing Accuracy', fontweight='bold')
ax1.set_xticks(x + width * 3)
ax1.set_xticklabels(dataset_names)
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax1.set_ylim(0, 1.1)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Plot 2: Train vs Test F1-Macro
ax2 = axes[1]
multiplier = 0
for model_name in models.keys():
    offset = width * multiplier * 2.5
    
    ax2.bar(x + offset, results_train_f1[model_name], width, label=f'{model_name} (Train)', alpha=0.6, hatch='//')
    ax2.bar(x + offset + width, results_test_f1[model_name], width, label=f'{model_name} (Test)')
    
    multiplier += 1

ax2.set_ylabel('F1-Macro Score')
ax2.set_title('Confronto Overfitting: Train F1-Macro vs Testing F1-Macro', fontweight='bold')
ax2.set_xticks(x + width * 3)
ax2.set_xticklabels(dataset_names)
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax2.set_ylim(0, 1.1)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

for data_name, (X_tr, y_tr) in datasets.items():
    print(f"--- Addestramento sul dataset: {data_name} ---")
    for model_name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        results_acc[model_name].append(acc)
        results_prec[model_name].append(prec)
        results_rec[model_name].append(rec)
        results_f1[model_name].append(f1)
        
        print(f"{model_name:20s} -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
    print()

# Plot Classificazione Supervisionata
x = np.arange(len(dataset_names))
width = 0.25

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics_to_plot = [
    ('Accuracy', results_acc, axes[0, 0]),
    ('Precision (Macro)', results_prec, axes[0, 1]),
    ('Recall (Macro)', results_rec, axes[1, 0]),
    ('F1-Macro', results_f1, axes[1, 1])
]

for title, metric_dict, ax in metrics_to_plot:
    multiplier = 0
    for model_name, scores in metric_dict.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, scores, width, label=model_name)
        ax.bar_label(rects, padding=3, fmt='%.3f', fontsize=9)
        multiplier += 1
        
    ax.set_ylabel('Score sul Test Set')
    ax.set_title(f'Task B: {title}', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(dataset_names)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()