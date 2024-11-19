import io
import pickle

import numpy as np
import time
import sklearn
import tensorflow as tf
import pandas as pd
import datetime
import keras
import pickle

from sklearn.preprocessing import MinMaxScaler

from celery import Celery , signals

from celery.result import AsyncResult
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback

import socketio
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Body, HTTPException, Form , Request

from fastapi.responses import JSONResponse
import aiofiles
import os
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Bidirectional, LSTM, Dense, Layer
from tensorflow.keras import backend as K
import tensorflow as tf
from keras import config
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Bidirectional, LSTM, Dense, Lambda, Flatten, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

import GPyOpt

tf.random.set_seed(42)
config.enable_unsafe_deserialization()


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],), initializer="zeros", trainable=True)
        super(Attention, self).build(input_shape)
        
    def call(self, x):
        et = K.tanh(K.dot(x, self.W) + self.b)
        at = K.softmax(et, axis=1)
        output = x * at
        return K.sum(output, axis=1)



class SelfAttention(Layer):
    def __init__(self, attention_dim):
        super(SelfAttention, self).__init__()
        self.attention_dim = attention_dim

    def build(self, input_shape):
        self.W_q = self.add_weight(shape=(input_shape[-1], self.attention_dim),
                                   initializer='random_normal', trainable=True)
        self.W_k = self.add_weight(shape=(input_shape[-1], self.attention_dim),
                                   initializer='random_normal', trainable=True)
        self.W_v = self.add_weight(shape=(input_shape[-1], self.attention_dim),
                                   initializer='random_normal', trainable=True)
        self.built = True

    def call(self, inputs):
        Q = tf.matmul(inputs, self.W_q)
        K = tf.matmul(inputs, self.W_k)
        V = tf.matmul(inputs, self.W_v)

        attention_scores = tf.matmul(Q, K, transpose_b=True)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        output = tf.matmul(attention_weights, V)

        return output

"""
###############################################    MODEL    #######################################################################################
###############################################    MODEL    #######################################################################################



# Enable unsafe deserialization
config.enable_unsafe_deserialization()

loaded_model____ = tf.keras.models.load_model("cnn_bi_lstm_uni_var_bo_model_8W_1H____.keras")


###############################################    VARS LOADING    #######################################################################################
###############################################    VARS LOADING    #######################################################################################



BATCH_SIZE_2048 = 2048
BATCH_SIZE_1024 = 1024
BATCH_SIZE_128 = 128
BATCH_SIZE_256 = 256
BATCH_SIZE_64 = 64
        
batch_size = BATCH_SIZE_1024

WINDOW_SIZE_400 = 400

WINDOW_SIZE_8 = 8
HORIZON_SIZE_1 = 1

WINDOW_SIZE_24 = 24
HORIZON_SIZE_4 = 4

WINDOW_SIZE = 8
HORIZON_SIZE = 1

horizon_size = HORIZON_SIZE
window_size = WINDOW_SIZE


combined_df = pd.DataFrame()
horizon_df = pd.DataFrame()

#with open('horizon_df (1).pkl', 'rb') as f:
#    horizon_df = pickle.load(f)

with open('scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
    
with open('scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)


with open('1_load_df.pkl', 'rb') as f:
    combined_df = pickle.load(f)



BATCH_SIZE_128 = 128
BATCH_SIZE_256 = 256
BATCH_SIZE_512 = 512
BATCH_SIZE_1024 = 1024
        
batch_size = BATCH_SIZE_128


WINDOW_SIZE_8 = 8
WINDOW_SIZE_24 = 24
WINDOW_SIZE_48 = 48

WINDOW_SIZE = WINDOW_SIZE_8 


"""

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# Create the Celery app instance
celery_app = Celery(
    'celery_app',
    broker='redis://127.0.0.1:6379/0',  # Redis as the broker
    backend='redis://127.0.0.1:6379/0'  # Redis as the result backend
)

# Update configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_concurrency=1,  # Limit to 1 task at a time
    task_acks_late=True,  # Ensure tasks are acknowledged only after they complete
    worker_prefetch_multiplier=1,
)


@signals.task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, **kwargs):
    print(f"Task {task_id} started")

@signals.task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, **kwargs):
    print(f"Task {task_id} completed")

@signals.task_success.connect
def task_success_handler(sender=None, result=None, **kwargs):
    print(f"Task {sender.name} with ID {sender.request.id} completed successfully.")
    print(f"Result: {result}")

@signals.task_failure.connect
def task_failure_handler(sender=None, exception=None, **kwargs):
    print(f"Task {sender.name} with ID {sender.request.id} failed.")
    print(f"Exception: {exception}")

sio = socketio.Client()


class ProgressCallback(Callback):
    def __init__(self, client_id, model_id):
        super().__init__()
        self.client_id = client_id
        self.model_id = model_id

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            progress_data = {
                'epoch': epoch,
                'loss': logs.get('loss'),
                'client_id': self.client_id,
                'model_id': self.model_id,
            }
            sio = socketio.Client()
            sio.connect('http://192.168.1.34:3056', wait_timeout=60)
            sio.emit('progress', progress_data)


class BayesianOptProgressCallback(Callback):
    def __init__(self, client_id, model_id, iteration):
        super().__init__()
        self.client_id = client_id
        self.model_id = model_id
        self.iteration = iteration
        
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            progress_data = {
                'epoch': epoch,
                'loss': logs.get('loss'),
                'client_id': self.client_id,
                'model_id': self.model_id,
                'iteration':self.iteration,
            }
            sio = socketio.Client()
            sio.connect('http://192.168.1.34:3056', wait_timeout=60)
            sio.emit('progressBayesian', progress_data)




def shifting_values(df, horizon_size, window_size):
    # Initialize a list to store DataFrames
    frames = [df.copy()]

    # Shifting values for each step in horizon_size
    for i in range(horizon_size):
        shifted_col = df['value'].shift(periods=-(i+window_size))
        frames.append(shifted_col.to_frame(f"V+{i+window_size}"))

    # Add windowed columns
    for i in range(window_size-1):
        shifted_col = df['value'].shift(periods=-(i+1))
        frames.append(shifted_col.to_frame(f"value+{i+1}"))

    # Concatenate all DataFrames in the list along the columns axis
    df = pd.concat(frames, axis=1)

    # Drop rows with NaN values
    df.dropna(inplace=True)
    
    return df


def y_horizon_df(df, horizon_size, window_size):
    columns_to_add = [f"V+{i+window_size}" for i in range(horizon_size)]
    horizon_df = df[columns_to_add].copy()
    df.drop(columns=columns_to_add, inplace=True)
    
    return df, horizon_df


def scaler_scaled_sets(X_set,y_set):
    # Initialize MinMaxScalers for X and y
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    # Fit the scalers on the training data
    scaler_X.fit(X_set)
    scaler_y.fit(y_set)

    # Transform the training data
    X_scaled = scaler_X.transform(X_set)
    y_scaled = scaler_y.transform(y_set)
    
    return scaler_X, scaler_y, X_scaled, y_scaled



def train_test_sets_split(X_scaled_set,y_scaled_set):
    split_size = int(len(X_scaled_set)*0.8)
    X_train, y_train = X_scaled_set[:split_size], y_scaled_set[:split_size]
    X_test, y_test = X_scaled_set[split_size:], y_scaled_set[split_size:]
    
    return X_train, y_train, X_test, y_test



def train_test_dataset_tensor_slices(X_train_set, y_train_set, X_test_set, y_test_set, batch_size = 1024):
    
    # Convert training and testing data to tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_set, y_train_set))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test_set, y_test_set))

    # Apply transformations: batch and prefetch
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    prediction_dataset = tf.data.Dataset.from_tensor_slices(X_test_set)
    prediction_dataset = prediction_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, test_dataset, prediction_dataset



def create_model_checkpoint(model_name, save_path="model_experiments"):
    # Modify the filepath to end in ".keras" as required
    filepath = os.path.join(save_path, f"{model_name}.keras")
    
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=filepath,  # create filepath to save model
        verbose=0,
        monitor='val_loss',  # only output a limited amount of text
        save_best_only=True  # save only the best model (according to monitor)
    )


def make_preds(model, input_data, scaler,horizon_size):

    forecast = model.predict(input_data)

    # Get the inference time
    if(horizon_size > 1):
        forecast = np.reshape(forecast, (-1, forecast.shape[-1]))
    else:
        forecast = forecast.reshape(-1,horizon_size)
        
    forecast = scaler.inverse_transform(forecast)
    return tf.squeeze(forecast)


def evaluate_preds(y_true, y_pred, scaler):

    # Make sure float32 (for metric calculations)
    y_true = scaler.inverse_transform(y_true)

    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # Calculate various metrics
    mae = keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = keras.metrics.mean_squared_error(y_true, y_pred) # puts and emphasis on outliers (all errors get squared)
    rmse = tf.sqrt(mse)
    mape = keras.metrics.mean_absolute_percentage_error(y_true, y_pred)

    # Account for different sized metrics (for longer horizons, reduce to single number)
    if mae.ndim > 0: # if mae isn't already a scalar, reduce it to one by aggregating tensors to mean
        mae = tf.reduce_mean(mae)
        mse = tf.reduce_mean(mse)
        rmse = tf.reduce_mean(rmse)
        mape = tf.reduce_mean(mape)
        


    return {"mae": mae.numpy().astype(float),
          "mse": mse.numpy().astype(float),
          "rmse": rmse.numpy().astype(float),
          "mape": mape.numpy().astype(float)}





def preparing_dataset(dataset_file, ):
    
    # create a dataframe from the csv file #### 
    # sort the dataframe
    # fill NaN or remove them
    # Lag Features (Sliding window features)
    # Extracting X and y data for the model to feed
    # scaling X and y
    # splitting train and test data
    # train the model
    # save model
    # save scalers
    # send paths
    # send evaluation metrics
    # send batch_size
    # send hyperparameters****X
    # 
    
    df = pd.read_csv("/kaggle/input/eliagriddataset/combined_df.csv", index_col='datetime', parse_dates=True)
    
    df = shifting_values(df,HORIZON_SIZE,WINDOW_SIZE)

    df, horizon_df = y_horizon_df(df, HORIZON_SIZE, WINDOW_SIZE)
    
    y = horizon_df.astype(np.float32)
    X = df.astype(np.float32)
    
    scaler_X, scaler_y, X_scaled, y_scaled = scaler_scaled_sets(X, y)

    X_train, y_train, X_test, y_test = train_test_sets_split(X_scaled,y_scaled)
    
    
    train_dataset, test_dataset, prediction_dataset = train_test_dataset_tensor_slices(X_train,y_train,X_test,y_test)
    
    
    
    
    
    return ''

def determine_window_size(horizon_size):
    switch = {
        1: 8,
        2: 10,
        3: 16,
        4: 24,
        5: 28,
        6: 28,
        7: 28,
        8: 32,
        9: 32,
        10: 32,
        11: 32,
        12: 34,
        13: 38,
        14: 38,
        15: 44,
        16: 48,
        24: 54,
    }

    # Get the window_size from the switch dictionary or calculate it
    window_size = switch.get(horizon_size, horizon_size * 2 if horizon_size > 16 else None)
    
    if window_size is None:
        raise ValueError("Invalid horizon_size")

    return window_size


def save_model_and_scalers(model, scaler_X, scaler_y, client_id, model_id):
    # Define base directory
    base_dir = os.path.join("saved_models", f"client_{client_id}", f"model_{model_id}")
    
    # Create directories if they don't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Paths for saving the model and scalers
    model_path = os.path.join(base_dir, f"{model.name}_{client_id}_{model_id}.keras")
    scaler_X_path = os.path.join(base_dir, f"scaler_X_{client_id}_{model_id}.pkl")
    scaler_y_path = os.path.join(base_dir, f"scaler_y_{client_id}_{model_id}.pkl")
    
    # Save the model
    model.save(model_path)
    
    # Save the scalers
    with open(scaler_X_path, 'wb') as f:
        pickle.dump(scaler_X, f)
    
    with open(scaler_y_path, 'wb') as f:
        pickle.dump(scaler_y, f)
    
    # Return the paths
    return {
        "model_path": model_path,
        "scaler_X_path": scaler_X_path,
        "scaler_y_path": scaler_y_path
    }




def create_model(learning_rate, lstm_units, kernel_1_size, kernel_2_size, dropout_1_rate,dropout_2_rate,dropout_3_rate, neuroPct, conv_1_filters, conv_2_filters, max_pool_1_size, max_pool_2_size, window_size, horizon_size, X_train):
    
    model = Sequential([
        Input(shape=(1,X_train.shape[2])),
        Conv1D(filters=conv_1_filters, kernel_size=kernel_1_size, padding='causal', activation='relu', input_shape=(1,X_train.shape[2])),
        MaxPooling1D(pool_size=max_pool_1_size, padding='same'),
        Conv1D(filters=conv_2_filters, kernel_size=kernel_2_size, padding='causal', activation='relu'),
        MaxPooling1D(pool_size=max_pool_2_size, padding='same'),
        # Keep the input shape 3D
        Bidirectional(LSTM(lstm_units, activation='relu')),
        Dense(neuroPct, activation='relu'),
        Dense(16, activation='relu'),
        # Output layer with HORIZON_SIZE
        Dense(horizon_size)
    ], name=f"cnn_bi_lstm_uni_var_bo_model_{window_size}W_{horizon_size}H")


    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mae')
    return model




UPLOAD_DIR = "./uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@celery_app.task(name="train_model_predefined_params",bind=True, max_retries=5)
def train_model_predefined_params(self, file_path: str, model_id: str, horizon_size: int, client_id:str):
    # Read CSV file into DataFrame
    #df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
    try:
        df = pd.read_csv(file_path, header=0, names=['datetime', 'value'])

        print(df)
        
        df['value'].fillna(df['value'].mean(), inplace=True)
        
        
        df.set_index('datetime', inplace=True)
        df.index = pd.to_datetime(df.index)
        
        df.sort_index(inplace=True)

        window_size = determine_window_size(horizon_size)
        
        df = shifting_values(df, horizon_size, window_size)  
        
        df, horizon_df = y_horizon_df(df, horizon_size, window_size)
        
        y = horizon_df.astype(np.float32)
        X = df.astype(np.float32)
        
        scaler_X, scaler_y, X_scaled, y_scaled = scaler_scaled_sets(X, y)
        
        X_train, y_train, X_test, y_test = train_test_sets_split(X_scaled, y_scaled)
        
        
        best_params = np.array([1.28000000e+02, 3.00000000e+00, 7.00000000e+00, 1.0240000e+03,
                            4.02558786e-04, 2.18375980e-01, 8.19128568e-03, 6.40000000e+01,
                            2.560000e+02, 5.12000000e+02, 2.00000000e+00, 3.20000000e+01,
                            5.11405730e-04])

        best_lstm_units = int(best_params[0])
        best_kernel_1_size = int(best_params[1])
        best_kernel_2_size = int(best_params[2])
        best_batch_size = int(best_params[3])
        best_dropout_1_rate = best_params[4]
        best_dropout_2_rate = best_params[5]
        best_dropout_3_rate = best_params[6]
        best_neuroPct = int(best_params[7])
        best_conv_1_filters = int(best_params[8])
        best_conv_2_filters = int(best_params[9])
        best_max_pool_1_size = int(best_params[10])
        best_max_pool_2_size = int(best_params[11])
        best_learning_rate = best_params[12]
        
        
        print( X_train.shape, y_train.shape,X_test.shape, y_test.shape )
        
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.values
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.values
            
        # Reshape the arrays
        X_train = X_train.reshape((-1, 1, X_train.shape[1]))
        y_train = y_train.reshape((-1, y_train.shape[1]))
        X_test = X_test.reshape((-1, 1, X_test.shape[1]))
        y_test = y_test.reshape((-1, y_test.shape[1]))
        

        train_dataset, test_dataset, prediction_dataset = train_test_dataset_tensor_slices(X_train, y_train, X_test, y_test, best_batch_size)



        model = create_model(best_learning_rate, best_lstm_units, best_kernel_1_size, best_kernel_2_size, best_dropout_1_rate, best_dropout_2_rate, best_dropout_3_rate,
                                        best_neuroPct, best_conv_1_filters, best_conv_2_filters,best_max_pool_1_size,best_max_pool_2_size, window_size, horizon_size, X_train)
        
        # Define EarlyStopping callback
        early_stopping = EarlyStopping(monitor='loss', patience=40, restore_best_weights=True)

        
        progress_callback = ProgressCallback(client_id=client_id, model_id=model_id)

        
        
        
        # Fit the model with validation data and callbacks
        model_history = model.fit(train_dataset,
                            epochs=200,
                            verbose=2,
                            validation_data=test_dataset,
                            callbacks=[early_stopping, progress_callback])
        
        
        model_preds = make_preds(model, prediction_dataset,scaler_y,horizon_size)
        
        
        if(horizon_size > 1):
            model_results = evaluate_preds(y_true=tf.squeeze(y_test),
                                        y_pred=tf.squeeze(model_preds),
                                        scaler=scaler_y)
        else:
            model_results = evaluate_preds(y_true=y_test,
                                            y_pred=tf.squeeze(model_preds),
                                            scaler=scaler_y)
        
        paths = save_model_and_scalers(model, scaler_X, scaler_y, client_id, model_id)
        
        sio = socketio.Client()
        sio.connect('http://192.168.1.34:3056', wait_timeout=60)
        sio.emit('task_complete', {
            "model_id": model_id,
            "client_id": client_id,
            "paths": paths,
            "window_size":window_size,
            "batch_size":best_batch_size,
            "model_results": model_results
        } )
        
    except Exception as e:
        print(f"Task failed: {e}")
        raise self.retry(exc=e, countdown=60)
    finally:
        # Cleanup code, if necessary
        print("Task completed")
    







@app.post("/train-forecasting-model")
async def train_forecasting_model(
    file: UploadFile = File(...), 
    model_id: str = Form(...), 
    horizon_size: int = Form(...),
    client_id: str = Form(...),
):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    print(f"hello {model_id}")
    try:
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail="File not saved correctly")

        #task = celery_app.send_task("train_model_predefined_params", args=[file_path, model_id, horizon_size, client_id])
        train_model_predefined_params.apply_async(args=[file_path, model_id, horizon_size, client_id], queue='training_queue')
        #train_model_predefined_params(file_path, model_id, horizon_size, client_id)

        return JSONResponse(
            status_code=200,
            content={"message": "File processed successfully", "model_id": model_id, "client_id":client_id}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
    """finally:
        if os.path.exists(file_path):
            os.remove(file_path)"""


# Define the objective function for GPyOpt
def objective_function(params, X_train,y_train,X_test,y_test, window_size, horizon_size, iteration_counter, client_id, model_id):
    
    iteration_counter['count'] += 1 
    
    lstm_units = int(params[0, 0])
    kernel_1_size = int(params[0, 1])
    kernel_2_size = int(params[0, 2])
    batch_size = int(params[0, 3])
    dropout_1_rate = params[0, 4]
    dropout_2_rate = params[0, 5]
    dropout_3_rate = params[0, 6]
    neuroPct = int(params[0, 7])
    conv_1_filters = int(params[0, 8])
    conv_2_filters = int(params[0, 9])
    max_pool_1_size = int(params[0, 10])
    max_pool_2_size = int(params[0, 11])
    learning_rate = params[0, 12]
    

    train_dataset, test_dataset, prediction_dataset = train_test_dataset_tensor_slices(X_train, y_train, X_test, y_test, batch_size)

    
    model = create_model(learning_rate, lstm_units, kernel_1_size, kernel_2_size, dropout_1_rate, dropout_2_rate, dropout_3_rate,
                                        neuroPct, conv_1_filters, conv_2_filters, max_pool_1_size, max_pool_2_size, window_size, horizon_size, X_train)
        
    

    """
    # Convert training and testing data to tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    # Apply transformations: batch and prefetch
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    prediction_dataset = tf.data.Dataset.from_tensor_slices(X_test)
    prediction_dataset = prediction_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    """
    
    # Define the EarlyStopping callback
    early_stopping = EarlyStopping(monitor='loss', patience=40, restore_best_weights=True)
    
    bayesianOptProgressCallback = BayesianOptProgressCallback(client_id=client_id, model_id=model_id, iteration=iteration_counter)

    # Train the model
    model.fit(train_dataset, epochs=200, verbose=2,
              validation_data=test_dataset,
              callbacks=[early_stopping, bayesianOptProgressCallback]) #create_model_checkpoint(model_name=(f"{model.name}")),
    
    # Evaluate the model
    loss = model.evaluate(test_dataset, verbose=0)
                         

                         
    return loss



@celery_app.task(name="train_bayesian_opt_model",bind=True, max_retries=5)
def train_bayesian_opt_model(self, file_path: str, model_id: str, horizon_size: int, client_id:str):
    try:
        df = pd.read_csv(file_path, header=0, names=['datetime', 'value'])

        print(df)
        
        df['value'].fillna(df['value'].mean(), inplace=True)
        
        
        df.set_index('datetime', inplace=True)
        df.index = pd.to_datetime(df.index)
        
        df.sort_index(inplace=True)

        window_size = determine_window_size(horizon_size)
        
        df = shifting_values(df, horizon_size, window_size)  
        
        df, horizon_df = y_horizon_df(df, horizon_size, window_size)
        
        y = horizon_df.astype(np.float32)
        X = df.astype(np.float32)
        
        scaler_X, scaler_y, X_scaled, y_scaled = scaler_scaled_sets(X, y)
        
        X_train, y_train, X_test, y_test = train_test_sets_split(X_scaled, y_scaled)
        
        
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.values
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.values
        
        # Reshape the arrays
        X_train = X_train.reshape((-1, 1, X_train.shape[1]))
        y_train = y_train.reshape((-1, y_train.shape[1]))
        X_test = X_test.reshape((-1, 1, X_test.shape[1]))
        y_test = y_test.reshape((-1, y_test.shape[1]))
    
        
        
        # Define the bounds for hyperparameters
        bounds = [
                {'name': 'lstm_units', 'type': 'discrete', 'domain': (128, 256)},
                {'name': 'kernel_1_size', 'type': 'discrete', 'domain': (3, 5)},
                {'name': 'kernel_2_size', 'type': 'discrete', 'domain': (3, 7)},
                {'name': 'batch_size', 'type': 'discrete', 'domain': (128,1024,2048)},
                {'name': 'dropout_1_rate', 'type': 'continuous', 'domain': (0.0, 0.01)},
                {'name': 'dropout_2_rate', 'type': 'continuous', 'domain': (0.0, 0.3)},
                {'name': 'dropout_3_rate', 'type': 'continuous', 'domain': (0.0, 0.01)},
                {'name': 'neuroPct', 'type': 'discrete', 'domain': (64, 128)},
                {'name': 'conv_1_filters', 'type': 'discrete', 'domain': (256, 1024)},
                {'name': 'conv_2_filters', 'type': 'discrete', 'domain': (512, 1024)},
                {'name': 'max_pool_1_size', 'type': 'discrete', 'domain': (2, 8)},
                {'name': 'max_pool_2_size', 'type': 'discrete', 'domain': (4, 32)},
                {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0004999, 0.0009999)}
        ]

        iteration_counter = {'count': 0}

        # Run the optimization
        optimizer = GPyOpt.methods.BayesianOptimization(
            f=lambda params: objective_function(params, X_train, y_train, X_test, y_test,window_size, horizon_size, iteration_counter, client_id, model_id),
            domain=bounds,
            acquisition_type='LCB',  # Use LCB acquisition function
            acquisition_weight=3.5,
            exact_feval=True,
            normalize_Y=False,
            maximize=False,  # Set maximize to False for minimization
            verbosity=True,  # Enable verbosity for logging optimization progress
            num_cores=4,
            seed=42
        )

        # Run the optimization
        optimizer.run_optimization(max_iter=2)

        # Get the best parameters
        best_params = optimizer.X[np.argmin(optimizer.Y)]
        print(f"Best hyperparameters: {best_params}")
        
        
        best_lstm_units = int(best_params[0])
        best_kernel_1_size = int(best_params[1])
        best_kernel_2_size = int(best_params[2])
        best_batch_size = int(best_params[3])
        best_dropout_1_rate = best_params[4]
        best_dropout_2_rate = best_params[5]
        best_dropout_3_rate = best_params[6]
        best_neuroPct = int(best_params[7])
        best_conv_1_filters = int(best_params[8])
        best_conv_2_filters = int(best_params[9])
        best_max_pool_1_size = int(best_params[10])
        best_max_pool_2_size = int(best_params[11])
        best_learning_rate = best_params[12]
        

        train_dataset, test_dataset, prediction_dataset = train_test_dataset_tensor_slices(X_train, y_train, X_test, y_test, best_batch_size)



        model = create_model(best_learning_rate, best_lstm_units, best_kernel_1_size, best_kernel_2_size, best_dropout_1_rate, best_dropout_2_rate, best_dropout_3_rate,
                                        best_neuroPct, best_conv_1_filters, best_conv_2_filters,best_max_pool_1_size,best_max_pool_2_size, window_size, horizon_size, X_train)
        
        # Define EarlyStopping callback
        early_stopping = EarlyStopping(monitor='loss', patience=40, restore_best_weights=True)

        
        progress_callback = ProgressCallback(client_id=client_id, model_id=model_id)

        
        
        
        # Fit the model with validation data and callbacks
        model_history = model.fit(train_dataset,
                            epochs=200,
                            verbose=2,
                            validation_data=test_dataset,
                            callbacks=[early_stopping, progress_callback])
        
        
        model_preds = make_preds(model, prediction_dataset,scaler_y,horizon_size)
        
        
        if(horizon_size > 1):
            model_results = evaluate_preds(y_true=tf.squeeze(y_test),
                                        y_pred=tf.squeeze(model_preds),
                                        scaler=scaler_y)
        else:
            model_results = evaluate_preds(y_true=y_test,
                                            y_pred=tf.squeeze(model_preds),
                                            scaler=scaler_y)
        
        paths = save_model_and_scalers(model, scaler_X, scaler_y, client_id, model_id)
        
        sio = socketio.Client()
        sio.connect('http://192.168.1.34:3056', wait_timeout=60)
        sio.emit('task_complete', {
            "model_id": model_id,
            "client_id": client_id,
            "paths": paths,
            "window_size":window_size,
            "batch_size":best_batch_size,
            "model_results": model_results
        } )
            
    except Exception as e:
        print(f"Task failed: {e}")
        raise self.retry(exc=e, countdown=60)
    finally:
        # Cleanup code, if necessary
        print("Task completed")    









"""

best_lstm_units = int(best_params[0])
best_kernel_1_size = int(best_params[1])
best_kernel_2_size = int(best_params[2])
best_batch_size = int(best_params[3])
best_dropout_1_rate = best_params[4]
best_dropout_2_rate = best_params[5]
best_dropout_3_rate = best_params[6]
best_neuroPct = int(best_params[7])
best_conv_1_filters = int(best_params[8])
best_conv_2_filters = int(best_params[9])
best_max_pool_1_size = int(best_params[10])
best_max_pool_2_size = int(best_params[11])
best_learning_rate = best_params[12]

def create_model(learning_rate, lstm_units, kernel_1_size, kernel_2_size, dropout_1_rate,dropout_2_rate,dropout_3_rate,
                 neuroPct, conv_1_filters, conv_2_filters, max_pool_1_size, max_pool_2_size):
    
    model = Sequential([
        #Input(shape=(1,X_train.shape[1])),
        Conv1D(filters=conv_1_filters, kernel_size=kernel_1_size, padding='causal', activation='relu', input_shape=(1,X_train.shape[1])),
        MaxPooling1D(pool_size=max_pool_1_size, padding='same'),
        #Dropout(dropout_1_rate),
        Conv1D(filters=conv_2_filters, kernel_size=kernel_1_size, padding='causal', activation='relu'),
        MaxPooling1D(pool_size=max_pool_1_size, padding='same'),
        #Dropout(dropout_2_rate),        
        # Keep the input shape 3D
        Bidirectional(LSTM(lstm_units, activation='relu')),
        #Dropout(dropout_3_rate),
        Dense(neuroPct, activation='relu'),
        Dense(16, activation='relu'),
        # Output layer with HORIZON_SIZE
        Dense(HORIZON_SIZE)
    ], name=f"cnn_bi_lstm_uni_var_bo_model_{WINDOW_SIZE}W_{HORIZON_SIZE}H")


    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mae')
    return model


def create_model_attention_mechanism_self_attention(learning_rate, lstm_units, kernel_1_size, kernel_2_size, dropout_1_rate,dropout_2_rate,dropout_3_rate,
                 neuroPct, conv_1_filters, conv_2_filters, max_pool_1_size, max_pool_2_size):
    
    model = Sequential([
        #Input(shape=(1,X_train.shape[1])),
        Conv1D(filters=conv_1_filters, kernel_size=kernel_1_size, padding='causal', activation='relu', input_shape=(1,X_train.shape[1])),
        MaxPooling1D(pool_size=max_pool_1_size, padding='same'),
        #Dropout(dropout_1_rate),
        Conv1D(filters=conv_2_filters, kernel_size=kernel_1_size, padding='causal', activation='relu'),
        MaxPooling1D(pool_size=max_pool_1_size, padding='same'),
        #Dropout(dropout_2_rate),        
        # Keep the input shape 3D
        SelfAttention(lstm_units),
        Bidirectional(LSTM(lstm_units, activation='relu',return_sequences=True)),
        Attention(),
        #Dropout(dropout_3_rate),
        Dense(neuroPct, activation='relu'),
        Dense(8, activation='relu'),
        # Output layer with HORIZON_SIZE
        Dense(HORIZON_SIZE)
    ], name=f"cnn_bi_lstm_uni_var_bo_model_{WINDOW_SIZE}W_{HORIZON_SIZE}H_create_model_attention_mechanism_regularization_clipping_gradient")


    optimizer = Adam(learning_rate=learning_rate)#, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='mae')
    return model


def create_model_attention_mechanism(learning_rate, lstm_units, kernel_1_size, kernel_2_size, dropout_1_rate,dropout_2_rate,dropout_3_rate,
                 neuroPct, conv_1_filters, conv_2_filters, max_pool_1_size, max_pool_2_size):
    
    model = Sequential([
        #Input(shape=(1,X_train.shape[1])),
        Conv1D(filters=conv_1_filters, kernel_size=kernel_1_size, padding='causal', activation='relu', input_shape=(1,X_train.shape[1])),
        MaxPooling1D(pool_size=max_pool_1_size, padding='same'),
        #Dropout(dropout_1_rate),
        Conv1D(filters=conv_2_filters, kernel_size=kernel_1_size, padding='causal', activation='relu'),
        MaxPooling1D(pool_size=max_pool_1_size, padding='same'),
        #Dropout(dropout_2_rate),        
        # Keep the input shape 3D
        Bidirectional(LSTM(lstm_units, activation='relu', return_sequences=True)),
        Attention(),
        #Dropout(dropout_3_rate),
        Dense(neuroPct, activation='relu'),
        Dense(16, activation='relu'),
        # Output layer with HORIZON_SIZE
        Dense(HORIZON_SIZE)
    ], name=f"cnn_bi_lstm_uni_var_bo_model_{WINDOW_SIZE}W_{HORIZON_SIZE}H")


    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mae')
    return model

"""




async def prepare_datasest_shapes():
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, pd.DataFrame):
        y_train = y_train.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.values
        
    # Reshape the arrays
    X_train_reshaped = X_train.reshape((-1, 1, X_train.shape[1]))
    y_train_reshaped = y_train.reshape((-1, 1, y_train.shape[1]))
    X_test_reshaped = X_test.reshape((-1, 1, X_test.shape[1]))
    y_test_reshaped = y_test.reshape((-1, 1, y_test.shape[1]))




async def train_model_attention():
    model = create_model_attention_mechanism(best_learning_rate, best_lstm_units, best_kernel_1_size, best_kernel_2_size, best_dropout_1_rate, best_dropout_2_rate, best_dropout_3_rate,
                                    best_neuroPct, best_conv_1_filters, best_conv_2_filters,best_max_pool_1_size,best_max_pool_2_size)


    # Convert training and testing data to tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_reshaped, y_train_reshaped))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test_reshaped, y_test_reshaped))

    # Apply transformations: batch and prefetch
    train_dataset = train_dataset.batch(best_batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(best_batch_size).prefetch(tf.data.AUTOTUNE)

    prediction_dataset = tf.data.Dataset.from_tensor_slices(X_test_reshaped)
    prediction_dataset = prediction_dataset.batch(best_batch_size).prefetch(tf.data.AUTOTUNE)

    early_stopping = EarlyStopping(monitor='val_loss', patience=90, restore_best_weights=True)
    start_time = time.time()
    model_history = model.fit(train_dataset,
                epochs=200,
                verbose=2,
                validation_data=test_dataset,
                callbacks=[create_model_checkpoint(model_name=f"final_model_test_{WINDOW_SIZE}W_{HORIZON_SIZE}H"),early_stopping])#, shuffle=False)
    end_time = time.time()



"""
X_train_fitted = X_train[len(X_train)-(int(len(X_train)/best_batch_size)*best_batch_size):]
y_train_fitted = y_train[len(y_train)-(int(len(y_train)/best_batch_size)*best_batch_size):]
X_test_fitted = X_test[:-(len(X_test)-(int(len(X_test)/best_batch_size)*best_batch_size))]
y_test_fitted = y_test[:-(len(y_test)-(int(len(y_test)/best_batch_size)*best_batch_size))]



if isinstance(X_train_fitted, pd.DataFrame):
    X_train_fitted = X_train_fitted.values
if isinstance(y_train_fitted, pd.DataFrame):
    y_train_fitted = y_train_fitted.values
if isinstance(X_test_fitted, pd.DataFrame):
    X_test_fitted = X_test_fitted.values
if isinstance(y_test_fitted, pd.DataFrame):
    y_test_fitted = y_test_fitted.values
    
# Reshape the arrays
X_train_reshaped = X_train_fitted.reshape((-1, 1, X_train_fitted.shape[1]))
y_train_reshaped = y_train_fitted.reshape((-1, 1, y_train_fitted.shape[1]))
X_test_reshaped = X_test_fitted.reshape((-1, 1, X_test_fitted.shape[1]))
y_test_reshaped = y_test_fitted.reshape((-1, 1, y_test_fitted.shape[1]))

async def train_model():
    final_model_test = create_model_attention_mechanism(best_learning_rate, best_lstm_units, best_kernel_1_size, best_kernel_2_size, best_dropout_1_rate, best_dropout_2_rate, best_dropout_3_rate,
                                    best_neuroPct, best_conv_1_filters, best_conv_2_filters,best_max_pool_1_size,best_max_pool_2_size)

    # Convert training and testing data to tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_reshaped, y_train_reshaped))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test_reshaped, y_test_reshaped))

    # Apply transformations: batch and prefetch
    train_dataset = train_dataset.batch(best_batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(best_batch_size).prefetch(tf.data.AUTOTUNE)

    prediction_dataset = tf.data.Dataset.from_tensor_slices(X_test_reshaped)
    prediction_dataset = prediction_dataset.batch(best_batch_size).prefetch(tf.data.AUTOTUNE)

    early_stopping = EarlyStopping(monitor='val_loss', patience=150, restore_best_weights=True)
    start_time = time.time()
    final_model_test_history = final_model_test.fit(train_dataset,
                epochs=200,
                verbose=2,
                validation_data=test_dataset,
                callbacks=[create_model_checkpoint(model_name=f"final_model_test_{WINDOW_SIZE}W_{HORIZON_SIZE}H"),early_stopping])#, shuffle=False)
    end_time = time.time()


"""




@app.get("/")
def read_root():
    return {"Hello": "World"}

"""
@app.post("/predict/")
async def pred(body):
    
    end_date = body['end_date']
    date_range = get_date_range(values, end_date) 
    into_future = get_into_future_steps(date_range)
    predictions = await make_future_forecasts(values, model_1, into_future, batch_size, scaler_X, scaler_y, WINDOW_SIZE)

    return {"predictions": predictions} 
"""


@app.post("/predict")
async def pred(request: List[float] = Body(...)):
    
    last_window = np.array(request).reshape(1, -1)
    
    last_window = scaler_X.transform(last_window)

    last_window_input = np.reshape(last_window, (1, 1, last_window.shape[1]))    
    
    future_forecast = loaded_model____.predict(last_window_input)

    future_forecast = future_forecast.reshape(-1, future_forecast.shape[-1])
    
    future_forecast = scaler_y.inverse_transform(future_forecast)
    
    prediction_result = {"prediction": future_forecast.tolist()[0][0]}
    
    return prediction_result 


@app.post("/forecast_into_the_future")
async def pred(request: Request ):
    try:
        
        data = await request.json()
        
        model_data = data.get("model_data")
        lastWindow = data.get("lastWindow")
        
        print(model_data, lastWindow)
        # get the last window
        # (optional use the batch size)
        # load the model
        # load the scalers
        # scale input & reshape
        # use model to predict
        # reshape result
        # scale the result
        # send back result
        
        user_model = tf.keras.models.load_model(model_data["model_path"])
        
        with open(model_data["scaler_X_path"], 'rb') as f:
            user_scaler_X = pickle.load(f)
        
        with open(model_data["scaler_y_path"], 'rb') as f:
            user_scaler_y = pickle.load(f)
            
        last_window = np.array(lastWindow).reshape(1, -1)
        
        last_window = user_scaler_X.transform(last_window)

        last_window_input = np.reshape(last_window, (1, 1, last_window.shape[1]))    
        
        future_forecast = user_model.predict(last_window_input)

        future_forecast = future_forecast.reshape(-1, future_forecast.shape[-1])
        
        future_forecast = user_scaler_y.inverse_transform(future_forecast)
        
        print("future_forecast",future_forecast)
        
        print("future_forecast.tolist()",future_forecast.tolist())

        prediction_result = {"prediction": future_forecast.tolist()[0]}
        
        return prediction_result 
    
    except Exception as e:
        print(f"Error while receiving data: {e}")
        return({"error": "Failed to process data"})



@app.post("/predict_test")
async def pred(request: List[float] = Body(...)):
    # Print the received data
    print(request)
    
    # Perform prediction (this is just a placeholder)
    prediction_result = {"prediction_test": "fjjgfgj"}
    
    return prediction_result




@app.post("/train-forecasting-model-bayesian")
async def train_forecasting_model_bayesian(
    file: UploadFile = File(...), 
    model_id: str = Form(...), 
    horizon_size: int = Form(...),
    client_id: str = Form(...),
):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    print(f"hello {model_id}")
    try:
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=500, detail="File not saved correctly")

        #task = celery_app.send_task("train_bayesian_opt_model", args=[file_path, model_id, horizon_size, client_id])
        train_bayesian_opt_model.apply_async(args=[file_path, model_id, horizon_size, client_id], queue='training_queue')
        #train_bayesian_opt_model(file_path, model_id, horizon_size, client_id)

        return JSONResponse(
            status_code=200,
            content={"message": "File processed successfully", "model_id": model_id, "client_id":client_id}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
    """finally:
        if os.path.exists(file_path):
            os.remove(file_path)"""
