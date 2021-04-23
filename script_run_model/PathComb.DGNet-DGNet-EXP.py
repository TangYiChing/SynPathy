"""
Perform Cross Validation
"""


import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

import tensorflow as tf
import shap as sp
import PathComb

import sklearn.preprocessing as skpre

def parse_parameter():
    parser = argparse.ArgumentParser(description='Validate Model with Cross Validation')

    parser.add_argument("-train", "--train_path",
                        default = "./data/TRAIN.pkl",
                        help = "path to TRAIN.pkl")
    parser.add_argument("-valid", "--valid_path",
                        default = "./data/VALID.pkl",
                        help = "path to VALID.pkl")
    parser.add_argument("-test", "--test_path",
                        default = "./data/TEST.pkl",
                        help = "path to TEST.pkl")
    parser.add_argument("-norm", "--normalization_str",
                        default = "tanh_norm",
                        choices = ["norm", "tanh", "tanh_norm", "standard", "minmax"],
                        help = "normalization methods")
    parser.add_argument("-g", "--gpu_str",
                        default = '0',
                        type = str,
                        help = "gpu device ids for CUDA_VISIBLE_DEVICES")
    parser.add_argument("-s", "--seed_int",
                        required = False,
                        default = 42,
                        type = int,
                        help = "seed for reproducibility. default=42")
    parser.add_argument("-m", '--train_test_mode', 
                        default=1, 
                        type = int,
                        help="Test of train mode (0: test, 1: train)")
    parser.add_argument("-wl", "--weightedLoss_bool",
                        default = False,
                        type = bool,
                        help = "ebanble weightedLoss for training if True")
    parser.add_argument("-shap", "--shap_bool",
                        default = False,
                        type = bool,
                        help = "enable SHAP if True")
    parser.add_argument('--gpu-support', default=True,
                        help='Use GPU support or not')
    parser.add_argument("-pretrained", '--saved_model_name', type=str,
                        help='Model name to save weights')
    parser.add_argument("-o", '--output_path', default="PathComb",
                        help='output prefix')
    return parser.parse_args()

def run_cv(args):
    """
    return prediction
    """
    ##################
    # load data
    #
    ##################
    train_df = pd.read_pickle(args.train_path)
    valid_df = pd.read_pickle(args.valid_path)
    test_df = pd.read_pickle(args.test_path)

    # get size of feature
    chem_col_list = [col for col in train_df.columns if col.startswith("CHEM")]
    dg_col_list = [col for col in train_df.columns if col.startswith("DGNet")]
    exp_col_list = [col for col in train_df.columns if col.startswith("EXP")]
    n_chem = len(chem_col_list)
    n_dg = len(dg_col_list)
    n_exp = len(exp_col_list)
    if n_chem + n_dg + n_exp + 1 < train_df.shape[1]:
        print("ERROR! #columns does not match")
        print("    #CHEM={:} | #DG-Net={:} | #EXP={:}".format(n_chem, n_dg, n_exp))
    else:
        print("    #CHEM={:} | #DG-Net={:} | #EXP={:}".format(n_chem, n_dg, n_exp))
    ##################
    # normalize data
    #
    ##################
    train_X_arr = train_df.iloc[:, 0:-1].values
    valid_X_arr = valid_df.iloc[:, 0:-1].values
    test_X_arr = test_df.iloc[:, 0:-1].values
    
    train_y_arr = train_df.iloc[:, -1].values
    valid_y_arr = valid_df.iloc[:, -1].values
    test_y_arr = test_df.iloc[:, -1].values

    chem_train_X_arr = train_df.iloc[:, :n_chem].values
    dg_train_X_arr = train_df.iloc[:, n_chem:n_chem+n_dg].values
    exp_train_X_arr = train_df.iloc[:, n_chem+n_dg:n_chem+n_dg+n_exp].values
    
    chem_valid_X_arr = valid_df.iloc[:, :n_chem].values
    dg_valid_X_arr = valid_df.iloc[:, n_chem:n_chem+n_dg].values
    exp_valid_X_arr = valid_df.iloc[:, n_chem+n_dg:n_chem+n_dg+n_exp].values

    chem_test_X_arr = test_df.iloc[:, :n_chem].values
    dg_test_X_arr = test_df.iloc[:, n_chem:n_chem+n_dg].values
    exp_test_X_arr = test_df.iloc[:, n_chem+n_dg:n_chem+n_dg+n_exp].values
    print("BEFORE NORMALIZATION")
    print("chem={:} | dg={:} | exp={:}".format(chem_train_X_arr.shape, dg_train_X_arr.shape, exp_train_X_arr.shape))
    print("chem={:} | dg={:} | exp={:}".format(chem_valid_X_arr.shape, dg_valid_X_arr.shape, exp_valid_X_arr.shape))
    print("chem={:} | dg={:} | exp={:}".format(chem_test_X_arr.shape, dg_test_X_arr.shape, exp_test_X_arr.shape))
    if args.normalization_str == "tanh_norm":
        # chem
        chem_train_X_arr, mean1, std1, mean2, std2, feat_filt = PathComb.normalize(chem_train_X_arr, norm="tanh_norm")
        chem_valid_X_arr, mmean1, sstd1, mmean2, sstd2, feat_filtt = PathComb.normalize(chem_valid_X_arr, mean1, std1, mean2, std2, feat_filt=feat_filt, norm="tanh_norm")
        chem_test_X_arr, mean1, std1, mean2, std2, feat_filt = PathComb.normalize(chem_test_X_arr, mean1, std1, mean2, std2, feat_filt=feat_filt, norm="tanh_norm")
        # dg
        dg_train_X_arr, mean1, std1, mean2, std2, feat_filt = PathComb.normalize(dg_train_X_arr,  norm="tanh_norm")
        dg_valid_X_arr, mmean1, sstd1, mmean2, sstd2, feat_filtt = PathComb.normalize(dg_valid_X_arr, mean1, std1, mean2, std2, feat_filt=feat_filt, norm="tanh_norm")
        dg_test_X_arr, mean1, std1, mean2, std2, feat_filt = PathComb.normalize(dg_test_X_arr, mean1, std1, mean2, std2, feat_filt=feat_filt, norm="tanh_norm")
        # exp
        exp_train_X_arr, mean1, std1, mean2, std2, feat_filt = PathComb.normalize(exp_train_X_arr,  norm="tanh_norm")
        exp_valid_X_arr, mmean1, sstd1, mmean2, sstd2, feat_filtt = PathComb.normalize(exp_valid_X_arr, mean1, std1, mean2, std2, feat_filt=feat_filt, norm="tanh_norm")
        exp_test_X_arr, mean1, std1, mean2, std2, feat_filt = PathComb.normalize(exp_test_X_arr, mean1, std1, mean2, std2, feat_filt=feat_filt, norm="tanh_norm")

    elif args.normalization_str == "tanh":
        # chem
        chem_train_X_arr, mean1, std1, feat_filt = PathComb.normalize(chem_train_X_arr, norm="tanh")
        chem_valid_X_arr, mmean1, sstd1, feat_filtt = PathComb.normalize(chem_valid_X_arr, mean1, std1, feat_filt=feat_filt, norm="tanh")
        chem_test_X_arr, mean1, std1, feat_filt = PathComb.normalize(chem_test_X_arr, mean1, std1,feat_filt=feat_filt, norm="tanh")
        # dg
        dg_train_X_arr, mean1, std1, feat_filt = PathComb.normalize(dg_train_X_arr, norm="tanh")
        dg_valid_X_arr, mmean1, sstd1, feat_filtt = PathComb.normalize(dg_valid_X_arr, mean1, std1, feat_filt=feat_filt, norm="tanh")
        dg_test_X_arr, mean1, std1, feat_filt = PathComb.normalize(dg_test_X_arr, mean1, std1,feat_filt=feat_filt, norm="tanh")
        # exp
        exp_train_X_arr, mean1, std1, feat_filt = PathComb.normalize(exp_train_X_arr, norm="tanh")
        exp_valid_X_arr, mmean1, sstd1, feat_filtt = PathComb.normalize(exp_valid_X_arr, mean1, std1, feat_filt=feat_filt, norm="tanh")
        exp_test_X_arr, mean1, std1, feat_filt = PathComb.normalize(exp_test_X_arr, mean1, std1, feat_filt=feat_filt, norm="tanh")

    elif args.normalization_str == "norm":
        # chem
        chem_train_X_arr, mean1, std1, feat_filt = PathComb.normalize(chem_train_X_arr, norm="norm")
        chem_valid_X_arr, mmean1, sstd1, feat_filtt = PathComb.normalize(chem_valid_X_arr, mean1, std1, feat_filt=feat_filt, norm="norm")
        chem_test_X_arr, mean1, std1, feat_filt = PathComb.normalize(chem_test_X_arr, mean1, std1, feat_filt=feat_filt, norm="norm")
        # dg
        dg_train_X_arr, mean1, std1, feat_filt = PathComb.normalize(dg_train_X_arr, norm="norm")
        dg_valid_X_arr, mmean1, sstd1, feat_filtt = PathComb.normalize(dg_valid_X_arr, mean1, std1, feat_filt=feat_filt, norm="norm")
        dg_test_X_arr, mean1, std1, feat_filt = PathComb.normalize(dg_test_X_arr, mean1, std1, feat_filt=feat_filt, norm="norm")
        # exp
        exp_train_X_arr, mean1, std1, feat_filt = PathComb.normalize(exp_train_X_arr, norm="norm")
        exp_valid_X_arr, mmean1, sstd1, feat_filtt = PathComb.normalize(exp_valid_X_arr, mean1, std1, feat_filt=feat_filt, norm="norm")
        exp_test_X_arr, mean1, std1, feat_filt = PathComb.normalize(exp_test_X_arr, mean1, std1, feat_filt=feat_filt, norm="norm")

    elif args.normalization_str == "standard":
        scaler = skpre.StandardScaler()
        # chem
        # dg
        dg_train_X_arr = scaler.fit_transform(dg_train_X_arr)
        dg_valid_X_arr = scaler.transform(dg_valid_X_arr)
        dg_test_X_arr = scaler.transform(dg_test_X_arr)
        # exp
        exp_train_X_arr = scaler.fit_transform(exp_train_X_arr)
        exp_valid_X_arr = scaler.transform(exp_valid_X_arr)
        exp_test_X_arr = scaler.transform(exp_test_X_arr)

    elif args.normalization_str == "minmax":
        scaler = skpre.MinMaxScaler()
        # chem
        # dg
        dg_train_X_arr = scaler.fit_transform(dg_train_X_arr)
        dg_valid_X_arr = scaler.transform(dg_valid_X_arr)
        dg_test_X_arr = scaler.transform(dg_test_X_arr)
        # exp
        exp_train_X_arr = scaler.fit_transform(exp_train_X_arr)
        exp_valid_X_arr = scaler.transform(exp_valid_X_arr)
        exp_test_X_arr = scaler.transform(exp_test_X_arr)

    else:
        print("ERROR! normalization method={:} not supported.".format(args.normalization_str))
        sys.exit(1)
    print("AFTER NORMALIZATION")
    print("chem={:} | dg={:} | exp={:}".format(chem_train_X_arr.shape, dg_train_X_arr.shape, exp_train_X_arr.shape))
    print("chem={:} | dg={:} | exp={:}".format(chem_valid_X_arr.shape, dg_valid_X_arr.shape, exp_valid_X_arr.shape))
    print("chem={:} | dg={:} | exp={:}".format(chem_test_X_arr.shape, dg_test_X_arr.shape, exp_test_X_arr.shape))
    ##################
    # initiate model
    #
    ##################
    inDrop_float = 0.2 # dropout for input layer
    dropout_float = 0.5
    #model = PathComb.generate_network_2(n_chem, n_dg, n_exp, inDrop_float, dropout_float)
    model = PathComb.generate_network_DGNet_DGNet_EXP_Xa(n_dg, n_exp, inDrop_float, dropout_float)
    print(model.summary())

    ##################
    # fit model
    #
    ##################
    l_rate = 0.0001
    max_epoch = 1000
    batch_size = 128
    earlyStop_patience = 25


    if args.train_test_mode == 1:
        modelName = args.output_path + ".saved_model.h5" # name of the model to save the weights
        print("TRAIN MODE: best model={:}".format(modelName))
        #############################################
        # define weightedLoss 
        # ref: https://github.com/tastanlab/matchmaker/blob/master/main.py
        #############################################
        if args.weightedLoss_bool == True:
            # calculate weights for weighted MSE loss
            min_s = np.amin(train_df['resp'].values)
            loss_weight = np.log(train_df['resp'].values - min_s + np.e)
            model = PathComb.dg_exp_trainer(model, l_rate, dg_train_X_arr, exp_train_X_arr, train_y_arr, dg_valid_X_arr, exp_valid_X_arr, valid_y_arr, max_epoch, batch_size,
                                    earlyStop_patience, modelName, weights=loss_weight)
        else:
            model = PathComb.dg_exp_trainer(model, l_rate, dg_train_X_arr, exp_train_X_arr, train_y_arr, dg_valid_X_arr, exp_valid_X_arr, valid_y_arr, max_epoch, batch_size,
                                    earlyStop_patience, modelName)


    else:

        modelName = args.saved_model_name
    print("AFTER TRAINING: modelName={:}".format(modelName))
    #############################
    # Test model with best model
    #############################
    print("modelName={:}".format(modelName))
    model.load_weights(modelName)
    prediction_arr = PathComb.predict(model, [dg_test_X_arr, exp_test_X_arr])

    prediction_df = test_df[['resp']].copy()
    prediction_df['prediction'] = prediction_arr.tolist()
    prediction_df.to_csv(args.output_path + '.PathComb.Prediction.txt', header=True, index=True, sep="\t")
    print(prediction_df)
    ####################
    # calcuate shapley
    ####################
    if args.shap_bool == True:
        print('    calculate shapely values')
        # random select 100 samples as baseline
        bg_dg = dg_train_X_arr[:100]
        bg_exp = exp_train_X_arr[:100]
        background = [bg_dg, bg_exp]
        explainer = sp.DeepExplainer(model, background)
        shap_list_list = explainer.shap_values([dg_test_X_arr, exp_test_X_arr])
        dg_shap_arr, exp_shap_arr = shap_list_list[0]
        shap_arr = np.concatenate( (dg_shap_arr, exp_shap_arr), axis=1)
        shap_df = pd.DataFrame(shap_arr, index=test_df.index, columns=test_df[dg_col_list+exp_col_list].columns)
        shap_df.to_csv(args.output_path + '.PathComb.SHAP.txt', header=True, index=True, sep="\t")
        print(shap_df)


    # return
    model.reset_states()
    return prediction_df

def cal_time(end, start):
    """return time spent"""
    # end = datetime.now(), start = datetime.now()
    datetimeFormat = '%Y-%m-%d %H:%M:%S.%f'
    spend = datetime.strptime(str(end), datetimeFormat) - \
            datetime.strptime(str(start),datetimeFormat)
    return spend

if __name__ == "__main__":
    # get args
    args = parse_parameter()
    # set seed
    np.random.seed(args.seed_int)
    tf.random.set_seed(args.seed_int)
    # set device
    num_cores = 8
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_str
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=10,
                                      inter_op_parallelism_threads=10,
                                      allow_soft_placement=True,
                                      device_count = {'CPU' : 1,
                                                      'GPU' : 1})
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
    # timer
    start = datetime.now()
    # execute function
    pred_df = run_cv(args)
    # save ouputs
    pred_df.to_csv(args.output_path+".PathComb.Prediction.txt", header=True, index=True, sep="\t")
    # finished
    print("[Finished in {:}]".format(cal_time(datetime.now(), start)))
