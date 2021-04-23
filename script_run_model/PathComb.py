"""
# Functions for PathComb

"""

import numpy as np

from tensorflow.keras import Sequential
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Concatenate, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.constraints import unit_norm, min_max_norm

def normalize(X, means1=None, std1=None, means2=None, std2=None, feat_filt=None, norm='tanh_norm'):
    """
    return normalized X

    ref: https://github.com/KristinaPreuer/DeepSynergy/blob/master/normalize.ipynb
    """
    if std1 is None:
        std1 = np.nanstd(X, axis=0)
    if feat_filt is None:
        feat_filt = std1!=0  # to avoid zero division
    X = X[:,feat_filt]
    X = np.ascontiguousarray(X)
    if means1 is None:
        means1 = np.mean(X, axis=0)
    X = (X-means1)/std1[feat_filt]
    if norm == 'norm':
        return(X, means1, std1, feat_filt)
    elif norm == 'tanh':
        return(np.tanh(X), means1, std1, feat_filt)
    elif norm == 'tanh_norm':
        X = np.tanh(X)
        if means2 is None:
            means2 = np.mean(X, axis=0)
        if std2 is None:
            std2 = np.std(X, axis=0)
        X = (X-means2)/std2
        X[:,std2==0]=0
        return(X, means1, std1, means2, std2, feat_filt)


def generate_network_DGNet_DGNet_EXP_Xa(n_dg, n_exp, inDrop_float, dropout_float):
    """
    return mdoel

    Note:
    =====
    dgnet --> FC2 --> enc_dgnet -|--> FC2 --> synergy
    exp   --> FC2 --> enc_exp   -|
    """
    ###
    # drug-drug features
    ###
    dg_input = Input( shape=(n_dg,) )

    ###
    # cell features
    ###
    exp_input = Input( shape=(n_exp,) )

    ###
    # combined: drug-drug-cell
    ###
    dg_exp_input = Concatenate()([dg_input, exp_input])

    dg_exp_hidden1 = Dense(8192, activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(0.00002))(dg_exp_input)
    dg_exp_hidden1 = Dropout(inDrop_float)(dg_exp_hidden1)
    dg_exp_hidden2 = Dense(4096, activation="relu", kernel_initializer="glorot_normal", kernel_regularizer=l2(0.00002))(dg_exp_hidden1)
    dg_exp_hidden2 = Dropout(dropout_float)(dg_exp_hidden2)
    dg_exp_output = Dense(1, activation="linear")(dg_exp_hidden2)

    # return model
    model = Model(inputs=[dg_input, exp_input], outputs=dg_exp_output)
    return model



def dg_exp_trainer(model, l_rate, dg_train_X, exp_train_X, train_y, dg_val_X, exp_val_X, val_y, epo, batch_size, earlyStop, modelName, weights=None):
    """
    return trained model
    """
    cb_check = ModelCheckpoint((modelName), verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=float(l_rate)), metrics=['mse','mae'])

    if weights is not None:
        model.fit([dg_train_X, exp_train_X], train_y, epochs=epo, shuffle=True, batch_size=batch_size,verbose=1,
                  validation_data=([dg_val_X, exp_val_X], val_y), sample_weight=weights,
                  callbacks=[EarlyStopping(monitor='val_loss', mode='auto', patience = earlyStop),cb_check])
    else:
        model.fit([dg_train_X, exp_train_X], train_y, epochs=epo, shuffle=True, batch_size=batch_size,verbose=1,
                  validation_data=([dg_val_X, exp_val_X], val_y),
                  callbacks=[EarlyStopping(monitor='val_loss', mode='auto', patience = earlyStop),cb_check])
    return model


def predict(model, data):
    """
    :param data: list of inputs
    """
    pred = model.predict(data)
    return pred.flatten()
