import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorboard.plugins.hparams import api as hp
import tensorflow_decision_forests as tfdf
import math

#train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('https://storage.googleapis.com/kagglesdsdata/competitions/34377/3220602/test.csv?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1676057779&Signature=S%2FUQRsgCFVvkSiZ1DiJUdmNeuVOyFQYKqQO9f4M2%2BMv%2F1DXmVaVxVMPEJ4mDmah4qi6aOhN8oY8owZFXy%2BBse6qLtJA7fmvy0RkNnUo1aN69TIDfbbOMI1WttJCC73n7NU66W3Qsnij%2FTPhd9sgFNbt3%2FlSTy5%2BTFUS3urOX7k71cViOOt9XVSa2zXLleDVI2%2BxQYnhVYYzYflEfJGos%2FzYT9ilivXkUsv%2BSeXfqm4Pd2Ja0x3aiqfdGL7L9e40gutkVwDXCjLR9%2Bb3JRsfoXuFl1lRw3%2FCz7HaXYDnR6DFdQ2XWW0W0fZ74eYaB65m8pDC1%2FfjUwuZ5RplNWINJtw%3D%3D&response-content-disposition=attachment%3B+filename%3Dtest.csv')
train_df = pd.read_csv('https://storage.googleapis.com/kagglesdsdata/competitions/34377/3220602/train.csv?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1676057811&Signature=e%2BGwleQwBsqiegw%2FPUNdwyG4OI%2F%2BBrMaV2gZZOGpvP5MXVvjxZkiv05uUiWeGsqXY3YODFRToCIlzct8RGUTTHJxA1QCsS2ZXVss82jr9%2BQSOcmaLm61rPu0AAO%2F2Dy9Nk3P0NsCCPAAo3WpreHk5SptBgM0FUFfuZjsabwPhUyHI209S47JRUDDkfOJ2pnpYe%2Bekw3SRfoVTRiJjXxvB1HnZjcHo5E9Mf7xDZYxl4e47GgcUOibjDR%2BvYKORlyZxfHzOIUNlubePG%2FmpoUVXdSr%2FdmbOZEpu%2FRd7OvirDB%2BVzVqYnxV2Kzmm7HQoy96btvKuRHUhSnL%2BMCXuLAq3w%3D%3D&response-content-disposition=attachment%3B+filename%3Dtrain.csv')

def data_preparation(df, val_split=0.1):
    #print(df.columns)
    #print(df.isnull().sum())
    #print(df.shape)

    df["HomePlanet"] = df["HomePlanet"].fillna("Earth")
    df["CryoSleep"] = df["CryoSleep"].fillna(False)
    #df["Cabin"] = df["Cabin"].fillna("S")
    df['Cabin'] = df['Cabin'].fillna("H/1900/C")
    df['Cabin'] = df['Cabin'].str.split(r"/")
    df[['c1', 'c2', 'c3']] = pd.DataFrame(df.Cabin.to_list())

    df["Destination"] = df["Destination"].fillna("TRAPPIST-1e")
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["VIP"] = df["VIP"].fillna(False)
    for col in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
        df[col] = df[col].fillna(df[col].median())
        # train_df["LuxuryClass"] += train_df[col]

    #y = df["Transported"].astype(int).to_numpy()
    #print(y_train)
    df = df.drop(
        ["PassengerId", "Name", "Cabin"], axis=1)
    # print(train_df)

    # binary hot encoding (true, false -> 1, 0)
    for i in ["CryoSleep", "VIP", "Transported"]:
        df[i] = df[i].astype(int)

    # print(train_df)

    # cabin encoding
    #df.loc[df["Cabin"].str[-1] == "P", "Cabin"] = 0
    #df.loc[df["Cabin"].str[-1] == "S", "Cabin"] = 1

    # print(train_df)
    # categories category values
    for header in ["HomePlanet", "Destination", "c1", "c2", "c3"]:
        values = list(df[header].value_counts().keys())
        #print(type(values))
        #print(values)
        for value in values:
            df[header] = df[header].replace(value, int(values.index(value)))

    #print(df)

    # categories numerical values

    df_train = df.iloc[:(round(df.shape[0] * (1 - val_split)))]
    df_val = df.iloc[(round(df.shape[0] * (1 - val_split))):]

    y_train = df_train["Transported"].to_numpy()
    y_val = df_val["Transported"].to_numpy()
    df_train = df_train.drop("Transported", axis=1)
    df_val = df_val.drop("Transported", axis=1)

    x_train = StandardScaler().fit_transform(df_train)
    x_val = StandardScaler().fit_transform(df_val)

    return x_train, y_train, x_val, y_val


x_train, y_train, x_val, y_val = data_preparation(train_df)

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([32, 64, 128]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.05, 0.1, 0.2]))
#HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete('adam'))

#LOG_DIR = './data/logs'

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )


def model_evaluation(hparams):
    model = tfk.Sequential([
        tfk.layers.Dense(hparams[HP_NUM_UNITS], activation='relu', input_shape=(13,)),
        tfk.layers.Dense(hparams[HP_NUM_UNITS], activation='relu'),
        tfk.layers.Dense(hparams[HP_NUM_UNITS], activation='relu'),
        #tfk.layers.Dense(hparams[HP_NUM_UNITS], activation='relu'),
        tfk.layers.Dropout(hparams[HP_DROPOUT]),
        tfk.layers.Dense(1)
    ])

    model.compile(
        optimizer=tfk.optimizers.Adam(),
        loss=tfk.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    model.fit(
        x_train,
        y_train,
        epochs=5,
        verbose=2,
        validation_data=(x_val, y_val),
        batch_size=32,
        callbacks=[
            #tfk.callbacks.TensorBoard(log_dir=LOG_DIR),
            #hp.KerasCallback(LOG_DIR, hparams=hparams)
        ]
    )
    _, acc = model.evaluate(x_val, y_val)
    return acc

session_num = 0

def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = model_evaluation(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


for num_units in HP_NUM_UNITS.domain.values:
    for dropout_rate in HP_DROPOUT.domain.values:
          hparams = {
              HP_NUM_UNITS: num_units,
              HP_DROPOUT: dropout_rate,
          }
          run_name = "run-%d" % session_num
          print('--- Starting trial: %s' % run_name)
          print({h.name: hparams[h] for h in hparams})
          run('logs/hparam_tuning/' + run_name, hparams)
          session_num += 1