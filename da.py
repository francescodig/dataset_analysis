import pandas as pd
import numpy as np

# Column names from allhypo.names
column_names = [
    'age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 'on_antithyroid_medication',
    'sick', 'pregnant', 'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
    'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych',
    'TSH_measured', 'TSH', 'T3_measured', 'T3', 'TT4_measured', 'TT4',
    'T4U_measured', 'T4U', 'FTI_measured', 'FTI', 'TBG_measured', 'TBG',
    'referral_source', 'target'
]

# Load and clean trailing ID
def load_data(path, col_names):
    df = pd.read_csv(path, header=None, names=col_names, na_values='?')
    # target has format "label.|ID" – strip the ID
    df['target'] = df['target'].str.split('|').str[0].str.strip()
    return df

df_train = load_data('allhypo.data', column_names)
df_test  = load_data('allhypo.test', column_names)

print("Train shape:", df_train.shape)
print("Test shape:", df_test.shape)
print("\nTarget distribution (train):")
print(df_train['target'].value_counts())
print("\nMissing values (train):")
print(df_train.isnull().sum()[df_train.isnull().sum() > 0])
print("\nDtypes sample:")
print(df_train.dtypes[:10])