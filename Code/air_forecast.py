import pandas as pd
from matplotlib import pyplot

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def read(path):
    # Load data
    df = pd.read_csv(path)
    df.drop('date', axis=1, inplace=True)
    print "Read data"
    print df.head()

    return df

def visualize(df):
    values = df.values
    print values
    # specify columns to plot
    groups = [1, 2, 3, 4, 6, 7, 8]
    i = 1
    # plot each column
    pyplot.figure()
    for group in groups:
    	pyplot.subplot(len(groups), 1, i)
    	pyplot.plot(values[:, group])
    	pyplot.title(df.columns[group], y=0.5, loc='right')
    	i += 1
    pyplot.show()
    # pyplot.savefig('Results/plots_air_pollution')

def encoding(values):
    encoder = LabelEncoder()
    values[:,4] = encoder.fit_transform(values[:,4])

    return values

def change_dtype(df):
    # print df.dtypes
    convert_dict = {'pollution': float,
                    'dew': float,
                    'wind_direction': float,
                    'hrs_snow': float,
                    'hrs_rain': float
               }
    df = df.astype(convert_dict)
    # print df.dtypes

    return df

def normalize_features(values):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    print "Normalized data"
    print scaled
    return scaled

'''
n_in: Number of lag observations as input (X).
        Values may be between [1..len(data)] Optional. Defaults to 1.
n_out: Number of observations as output (y).
        Values may be between [0..len(data)-1]. Optional. Defaults to 1.
'''
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    print type(data)
    n_vars = 1 if type(data) is list else data.shape[1]
    print n_vars
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
    	cols.append(df.shift(i))
    	names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
    	cols.append(df.shift(-i))
    	if i == 0:
    		names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    	else:
    		names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
    	agg.dropna(inplace=True)

    return agg

def split(data):
    values = data.values
    n_train_hours = 365 * 24

    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]

    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    print train_X.shape, train_y.shape, test_X.shape, test_y.shape

    return train_X, train_y, test_X, test_y

def network(train_X, train_y, test_X, test_y):
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72,
                        validation_data=(test_X, test_y), verbose=2, shuffle=False)

    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

    # make a prediction
    # yhat = model.predict(test_X)
    # test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

def main():
    path = 'Data/pollution.csv'
    data = read(path)
    # visualize(data)
    values = data.values
    # print values
    values = encoding(values)
    # data = change_dtype(data)
    values = values.astype('float32')
    print values
    scaled= normalize_features(values)

    # convert series to supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    print reframed.head()

    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
    print reframed.head()

    # split into train and test sets
    train_X, train_y, test_X, test_y = split(reframed)

    network(train_X, train_y, test_X, test_y)

if __name__ == '__main__':
    main()
