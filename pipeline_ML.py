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

df_train = load_data('allhypo.data')
df_test  = load_data('allhypo.test')

# Drop TBG (entirely missing)
df_train = df_train.drop(columns=['TBG', 'TBG_measured'])
df_test  = df_test.drop(columns=['TBG', 'TBG_measured'])

# Binary encode f/t columns
binary_cols = ['on_thyroxine','query_on_thyroxine','on_antithyroid_medication',
               'sick','pregnant','thyroid_surgery','I131_treatment','query_hypothyroid',
               'query_hyperthyroid','lithium','goitre','tumor','hypopituitary','psych',
               'TSH_measured','T3_measured','TT4_measured','T4U_measured','FTI_measured']
for c in binary_cols:
    df_train[c] = (df_train[c] == 't').astype(int)
    df_test[c]  = (df_test[c]  == 't').astype(int)

# One-hot encoding for sex and referral_source
df_train = pd.get_dummies(df_train, columns=['sex', 'referral_source'], drop_first=False)
df_test  = pd.get_dummies(df_test,  columns=['sex', 'referral_source'], drop_first=False)
df_train, df_test = df_train.align(df_test, join='left', axis=1, fill_value=0)

# Target encode
y_train = df_train['target']
y_test  = df_test['target']
X_train = df_train.drop(columns=['target'])
X_test  = df_test.drop(columns=['target'])

# Outlier removal (3 sigma on numeric cols)
num_cols = ['age','TSH','T3','TT4','T4U','FTI']
for c in num_cols:
    mean, std = X_train[c].mean(), X_train[c].std()
    mask = (X_train[c] - mean).abs() <= 3 * std
    X_train = X_train[mask]
    y_train = y_train[mask]
print("After outlier removal:", X_train.shape)

# Median imputer
imp = SimpleImputer(strategy='median')
X_train_imp = imp.fit_transform(X_train)
X_test_imp  = imp.transform(X_test)

# Standard scaler
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_imp)
X_test_sc  = scaler.transform(X_test_imp)

# Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=42),
}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    pred = model.predict(X_test_sc)
    print(f"{name}: {accuracy_score(y_test, pred):.4f}")

# KMeans
km = KMeans(n_clusters=4, random_state=42, n_init=10)
km.fit(X_train_sc)
print("KMeans done, clusters:", np.unique(km.labels_))