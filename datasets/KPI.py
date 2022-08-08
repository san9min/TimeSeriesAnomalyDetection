import torch
import pandas as pd 
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import minmax_scale

from util.sliding_window import sliding_window

def build_kpi(args):
    train, test = KPI_Dataprocessing(args)
    
    train = sliding_window(train, args)  # sliding window
    test = sliding_window(test, args)

    train = KPIDataset(train)
    test = KPIDataset(test)

    return train, test

class KPIDataset(Dataset):

    def __init__(self, dataset):
        super(KPIDataset,self).__init__()

        self.timestamp = dataset['timestamp']
        self.value = dataset['value']
        self.label = dataset['label']

    def __len__(self):
        return len(self.timestamp)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        timestamp = self.timestamp[idx]
        value = self.value[idx]
        label = self.label[idx]

        time_stamp = np.array(timestamp)
        value = np.array(value) 
        label = np.array(label)

        return time_stamp, value, label


def KPI_Dataprocessing(args):

    kpi_id = ['da10a69f-d836-3baa-ad40-3e548ecf1fbd',
    'e0747cad-8dc8-38a9-a9ab-855b61f5551d',
    'ab216663-dcc2-3a24-b1ee-2c3e550e06c9',
    '54350a12-7a9d-3ca8-b81f-f886b9d156fd',
    'a8c06b47-cc41-3738-9110-12df0ee4c721',
    '0efb375b-b902-3661-ab23-9a0bb799f4e3',
    'c02607e8-7399-3dde-9d28-8a8da5e5d251',
    '301c70d8-1630-35ac-8f96-bc1b6f4359ea',
    '7103fa0f-cac4-314f-addc-866190247439',
    '4d2af31a-9916-3d9f-8a8e-8a268a48c095',
    '6a757df4-95e5-3357-8406-165e2bd49360',
    'ffb82d38-5f00-37db-abc0-5d2e4e4cb6aa',
    '57051487-3a40-3828-9084-a12f7f23ee38',
    'f0932edd-6400-3e63-9559-0a9860a1baa9',
    '431a8542-c468-3988-a508-3afd06a218da',
    '1c6d7a26-1f1a-3321-bb4d-7a9d969ec8f0',
    'c69a50cf-ee03-3bd7-831e-407d36c7ee91',
    '05f10d3a-239c-3bef-9bdc-a2feeb0037aa',
    '847e8ecc-f8d2-3a93-9107-f367a0aab37d',
    '6efa3a07-4544-34a0-b921-a155bd1a05e8',
    '43115f2a-baeb-3b01-96f7-4ea14188343c',
    '9c639a46-34c8-39bc-aaf0-9144b37adfc8',
    'a07ac296-de40-3a7c-8df3-91f642cc14d0',
    'ba5f3328-9f3f-3ff5-a683-84437d16d554',
    '55f8b8b8-b659-38df-b3df-e4a5a8a54bc9',
    '6d1114ae-be04-3c46-b5aa-be1a003a57cd',
    '8723f0fb-eaef-32e6-b372-6034c9c04b80',
    'adb2fde9-8589-3f5b-a410-5fe14386c7af',
    '42d6616d-c9c5-370a-a8ba-17ead74f3114']
    
    num_episodes = len(kpi_id)
    split_bar = int(num_episodes * args.split_ratio)

    df = pd.read_csv(args.data_path + "/AIOps/KPI.csv")

    train = []
    test = []
    for i, id in enumerate(kpi_id) :
        if i < split_bar: #train
            df_i = df[df['KPI ID'] == id]
            train.append({
                'timestamp' : df_i['timestamp'].tolist(),
                'value' : minmax_scale(df['value'].tolist()),
                'label': df_i['label'].tolist()
            })
        else : #test
            df_i = df[df['KPI ID'] == id]
            test.append({
                'timestamp': df_i['timestamp'].tolist(),
                'value': minmax_scale(df_i['value'].tolist()),
                'label': df_i['label'].tolist()
                })

    return train, test