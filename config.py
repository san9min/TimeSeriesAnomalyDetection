import argparse
import torch

# Params
def get_parse():
    args = argparse.Namespace()

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #data
    args.datasets = 'Yahoo'  # Yahoo, SWaT, Numenta, KPI
    args.using_data = 'A1'
    args.data_path = 'dataset/'
    args.window_size = 25
    args.split_ratio = 0.8

    #prtrain
    args.pretrain = False
    args.pre_trained_weights = 'Check-point_A1'

    #reward
    args.TP = 1.0
    args.TN = 2e-1
    args.FP = -2e-1
    args.FN = -1.0

    #buffer
    args.batch_size = 300
    args.capacity = 1000
    args.anomaly_ratio = 0.3

    #metric
    args.beta = 1

    #ICM
    args.lambda_ = 0.1 #Q loss
    args.beta_ = 0.2  #inverse, forward loss
    args.eta = 1.0

    #loss
    args.qscale = 1e5
    args.fscale = 1.
    args.iscale = 1e4

    # eps greedy policy
    args.eps = 1.0
    args.eps_decay = 0.98

    #train
    args.epochs = 2000
    args.gamma = 0.8 # for Q networ train
    args.lr = 1e-3

    return args    

