import tensorflow as tf
import pandas as pd
import numpy as np

import time

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from random import randint
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')
train_features = ['ws_10m', 'GHI']
hours_ahead = 6
window_length = int(hours_ahead * 60/10)
batch_size = 256
n_features = 1

results = {feat: {'history': [], 'model': []} for feat in train_features}
scalers = {feat: MinMaxScaler() for feat in train_features}


def generate_time_series(x_train, y_train, batch_size, window_length):
    ts_train = TimeseriesGenerator(
        data=x_train,
        targets=y_train,
        batch_size=batch_size,
        length=window_length
    )

    ts_test = TimeseriesGenerator(
        data=x_test,
        targets=y_test,
        batch_size=batch_size,
        length=window_length
    )

    return ts_train, ts_test


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(
            64,
            input_shape=(window_length, n_features),
            return_sequences=True
        ),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.SimpleRNN(
            32,
            input_shape=(window_length, n_features),
            return_sequences=False
        ),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    model.compile(
        loss=tf.losses.MeanSquaredError(),
        optimizer=tf.optimizers.Adam(),
        metrics=['mean_absolute_error']
    )

    return model


def create_prediction_dataframe(ts_test, y_test, window_length, timesteps, df):
    test_pred = model.predict(ts_test)

    y = scalers[feature_name].inverse_transform(y_test)
    y_hat = scalers[feature_name].inverse_transform(test_pred)

    lag = window_length + timesteps+1

    return pd.DataFrame(
        {'y': y[lag:, 0], 'y_hat': y_hat[timesteps+1:, 0]},
        index=df.index[int(0.75*len(df))+lag:]
    )


def plot_sample(nn, ylabel="", p=0.1):
    n = int(len(nn) * p)

    ini = randint(0, len(nn)-n)

    nn.iloc[ini:ini+n].plot(ylabel=ylabel)


def last_rmse(results):
    rmse = {}

    n = 1
    for feature, data in results.items():
        rmse[feature] = []

        last_rmse = None
        for history in data['history']:
            last_rmse = history['val_loss'].iloc[-1]
            rmse[feature].append(last_rmse)

        n = len(rmse[feature]) + 1

    return pd.DataFrame(rmse, index=range(1, n))


def compare_prediction(features_results):

    window_length = len(features_results['model'])
    sample = features_results['model'][0]
    length = len(sample)
    start = randint(1, length - window_length)
    start_pred = window_length + start

    values = sample[start:start_pred].copy(deep=True)
    values['y_hat'] = None

    for nn in features_results['model']:
        values = values.append(nn.iloc[start_pred])

    return values


if __name__ == "__main__":

    for feature_name in train_features:
        print(f'feature name: {feature_name}')

    values = df[feature_name].values.reshape(-1, 1)
    features = scalers[feature_name].fit_transform(values)

    for timesteps in range(window_length):
        i = time.time()
        print(f'timestep: {timesteps}')

        labels = np.roll(features, -timesteps)

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=0.25,
            shuffle=False
        )

        ts_train, ts_test = generate_time_series(
            x_train,
            y_train,
            batch_size,
            window_length
        )

        model = create_model()

        history = model.fit(
            ts_train,
            validation_data=ts_test,
            callbacks=early_stopping,
            epochs=100,
            shuffle=False,
            verbose=0
        )

        loss = history.history['val_loss']
        rmse = loss[-1]
        print(f'RMSE: {rmse}')
        print(f'epochs: {len(loss)}')

        history_df = pd.DataFrame(history.history)
        nn = create_prediction_dataframe(
            ts_test,
            y_test,
            window_length,
            timesteps,
            df
        )

        results[feature_name]['model'].append(nn.copy(deep=True))
        results[feature_name]['history'].append(history_df.copy(deep=True))

        f = time.time()
        print(f'{f - i:.3} s')
        print('-----')
    print()
