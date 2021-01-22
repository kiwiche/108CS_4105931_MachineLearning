from sklearn import preprocessing
import pandas as pd
import numpy as np

def normalize_data(df):
    df.drop(['symbol'],axis=1, inplace=True)
    min_max_scaler = preprocessing.MinMaxScaler()
    df['open'] = min_max_scaler.fit_transform(df.open.values.reshape(-1,1))
    df['close'] = min_max_scaler.fit_transform(df.close.values.reshape(-1,1))
    df['high'] = min_max_scaler.fit_transform(df.high.values.reshape(-1,1))
    df['low'] = min_max_scaler.fit_transform(df.low.values.reshape(-1,1))
    df['volume'] = min_max_scaler.fit_transform(df.volume.values.reshape(-1,1))
    return df

def data_split(stock, seq_len=15):
    amount_of_features = len(stock.columns) # 5
    # stock = stock.as_matrix()
    print(stock.shape)
    sequence_length = seq_len + 1 # index starting from 0
    result = []

    for i in range(stock.shape[0] - sequence_length): # maxmimum date = lastest date - sequence length
        result.append(np.array(stock[i: i + sequence_length])) # index : index + 15days

    result = np.array(result)
    row = round(0.85 * result.shape[0]) # 85% split
    train = result[:int(row), :] # 85% date, all features 
    
    x_train = train[:, :-1] 
    y_train =np.array(train[:, -1][:,1])
    
    x_test = result[int(row):, :-1] 
    y_test = np.array(result[int(row):, -1][:,1])

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

    return [x_train, y_train, x_test, y_test]

def shuffle(x_train, y_train):
    np.random.seed(10)
    random_list = np.arange(x_train.shape[0])
    np.random.shuffle(random_list)
    return x_train[random_list], y_train[random_list]
    