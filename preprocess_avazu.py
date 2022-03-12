import numpy as np
import pandas as pd
import random
import pickle

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


avazu_file_path = r'/dataset/avazu_data/train'
df = pd.read_table(avazu_file_path, sep=',')


ratio = 1.0
num_rows = df.shape[0]
print('finish reading data for {} entries'.format(num_rows))

chosen_index = np.random.choice(num_rows, replace=False, size = int(num_rows * ratio))
df = df.iloc[chosen_index]
print('finish preparing indices')

features = df.columns.tolist()
df[features] = df[features].fillna('',)

print (features)

# # label encoding for categorical features
label_encoder_dict = {}
for feat in features:
    lbe = LabelEncoder() # encode target labels with value between 0 and n_classes-1.
    df.loc[:,feat] = lbe.fit_transform(df[feat]) # fit label encoder and return encoded label
    df.loc[:,feat] = df[feat].astype(np.int32) # convert from float64 to float32
    label_encoder_dict[feat] = lbe # store the fitted label encoder

train_data, test_data = train_test_split(df, test_size=0.1, random_state=42)


for key in features:
    print(key)
    print('train', train_data[key].nunique())
    print('test', test_data[key].nunique())

vocab_dict = {}
for feat in features:
    vocab_dict[feat] = df[feat].nunique()

with open('/dataset/avazu_data/vocab_size_{}.pickle'.format(ratio), 'wb') as handle:
    pickle.dump(vocab_dict, handle)

train_data.to_csv(path_or_buf='/dataset/avazu_data/new_train_{}_0.9.csv'.format(ratio), index=False)
test_data.to_csv(path_or_buf='/dataset/avazu_data/new_test_{}_0.1.csv'.format(ratio), index=False)