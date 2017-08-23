import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_file(file_name):
    dataset = pd.read_csv("{0}.csv".format(file_name));
    # dataset = pd.read_csv('{0}.csv'.format(file_name), dtype={'query_id': 'category', 'url_id': 'category','is_homepage': 'category',
    #                     'sig3': 'float64', 'sig4': 'float64', 'sig5': 'float64', 'sig6': 'float64'})

    return dataset


def plus_1(x): return x + 1


test_data = read_file("test")

train_data = read_file("training")

train_data['sig3'] = train_data['sig3'].apply(plus_1)
train_data['sig4'] = train_data['sig4'].apply(plus_1)
train_data['sig5'] = train_data['sig5'].apply(plus_1)
train_data['sig6'] = train_data['sig6'].apply(plus_1)

train_data['sig3'] = train_data['sig3'].apply(np.log)
train_data['sig4'] = train_data['sig4'].apply(np.log)
train_data['sig5'] = train_data['sig5'].apply(np.log)
train_data['sig6'] = train_data['sig6'].apply(np.log)

print(train_data.dtypes)

group = train_data.groupby('query_id', as_index=False)['url_id'].count()
print(type(group))

# for col in ['query_id', 'url_id', 'is_homepage']:
#   train_data[col] = train_data[col].astype('category')





# Handling only training data from now
X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]

X_new = np.log(1 + X['sig3'])

# Transformation
# sig3 - 6
X[:, 6] = np.log(1 + X[:, 6])
X[:, 7] = np.log(1 + X[:, 7])
X[:, 8] = np.log(1 + X[:, 8])
X[:, 9] = np.log(1 + X[:, 9])





















