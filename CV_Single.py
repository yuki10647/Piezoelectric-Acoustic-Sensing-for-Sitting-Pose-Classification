import torch
import torchvision.transforms as transforms
import scipy.io.wavfile
import torch.nn as nn
import torch.nn.functional as F
import librosa
import librosa.display
import numpy as np
import pandas as pd
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import os
"""
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda")
"""
device = torch.device("cpu")
#people_num = 1 # 被験者数
LABEL_SIZE = 12 # classの数
BATCH_SIZE = 4  # <-- 改善の余地あり
EPOCH = 30 # 増やす余地あり
INF = 100000000000
KFold = 6


classes = ('普通',
           '右手挙げ',
           '左手挙げ',
           '両手挙げ',
           '右手すり',
           '左手すり',
           '両手すり',
           '背もたれ',
           '前傾姿勢',
           '浅く座る',
           '右足組',
           '左足組')

 
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

"""
8*513*87 121
12*509*83 117
12*254*41 58
16*250*37 54 36000 54000 
"""

class RealNet(nn.Module):
    def __init__(self):# 変える箇所1：ネットワークのサイズ
        super().__init__()
        self.conv1 = nn.Conv2d(8, 12, 5) # in_channels, out_channels, kernel_size
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 16, 5)
        self.fc1 = nn.Linear(36000, 1000)
        self.fc3 = nn.Linear(1000, LABEL_SIZE)#出力サイズ#TODO:LABEL_SIZEを識別するクラス数に変更する。
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x_1 = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x_1))
        x = self.fc3(x)
        return x, x_1
    
    

def FILE2DATA(files):
    datas = []
    rates = []
    file_name = []
    for i in range(len(files)):
        amps = []
        for file in files.iloc[i, :]:
            amps.append(file)
        datas.append(i)
        file_name.append(amps)
    return file_name, 44100, np.array(datas)

def STFT(files):
    stft_data=[]
    fft_size = 1024
    hop_length = int(fft_size / 4)
            
    for j in range(np.shape(files)[1]):
        mic_pair=[]
        for i in range(np.shape(files)[0]):
            rate, data = scipy.io.wavfile.read(files[i][j])
            data = data / 32768
            """
            # 0.1秒分をトリミング（サンプリングレートに合わせて調整する）
            trim_length = int(0.15 * rate)
            data = data[trim_length:]
            """
            # 0.25以降をトリミング
            trim_length2 = int(0.7 * rate)
            data = data[:trim_length2]       
            
            amplitude = np.abs(librosa.core.stft(
                data, n_fft=fft_size, hop_length=hop_length))
            amplitude=torch.from_numpy(amplitude)
            mic_pair.append(np.array(amplitude))
                
        stft_data.append(mic_pair)
    stft_data = np.squeeze(np.array(stft_data))
    return np.array(stft_data)



class MyDataset(torch.utils.data.Dataset):
    def __init__(self,data_size=INF,dt=None, transform=None,out1vec=True):
        df = dt
        labels = df.iloc[:,0].to_numpy() 
        audio_data = df.iloc[:,2:10]    # 2:6 or 2:10
        files,_,_ = FILE2DATA(audio_data)
        self.transform = transform
        self.data_num = len(audio_data)

        self.data = []
        self.label = []

        for file in files:
            self.data.append(file)
        for label in labels:
            self.label.append(label)
        
    def __len__(self):
        return self.data_num
    def __getitem__(self, idx):
        out_data = self.data[idx]
        out_label = self.label[idx]
        return out_data, out_label


count_matrix = np.zeros((EPOCH, len(classes), len(classes))) # (50, 12, 12, )
acc = np.zeros((EPOCH,len(classes))) # (50, 12, )


df_all = pd.read_csv('csv/TrainSet_RE2.csv',header=None)
df_all_swp = pd.read_csv('csv/TrainSet_RE2_swp_re.csv',header=None)
labels = df_all.iloc[:,0].to_numpy()

skf = StratifiedKFold(n_splits=KFold)
for train_index, test_index in skf.split(df_all, labels):
    print("start Fold")

    train_df = df_all.iloc[train_index,:]
    train_swp = df_all_swp.iloc[train_index,:]                  # ChannelSwap使用
    print(train_df.shape, train_swp.shape)
    test_df = df_all.iloc[test_index,:]
    print(test_index[0])

    trainset1=MyDataset(dt=train_df, transform=transforms.ToTensor(),out1vec=False)
    trainloader1=torch.utils.data.DataLoader(trainset1, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    trainset2=MyDataset(dt=train_swp, transform=transforms.ToTensor(),out1vec=False)
    trainloader2=torch.utils.data.DataLoader(trainset2, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    testset=MyDataset(dt=test_df, transform=transforms.ToTensor(),out1vec=False)
    testloader=torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    net1 = RealNet()
    net1.to(device)

    criterion = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    optimizer1 = optim.SGD(net1.parameters(), lr=0.001, momentum=0.9)
    scheduler1 = StepLR(optimizer1, step_size=10, gamma=0.5)

    for epoch in tqdm(range(EPOCH),desc="User"):

        ########## train data #####################
        for data1, data2 in zip(trainloader1, trainloader2):
            # trainloader1から収音データ取得
            inputs1, labels1 = data1[0], data1[1].to(device)
            inputs1 = STFT(inputs1)
            inputs1 = torch.from_numpy(inputs1.astype(np.float32))
            inputs1 = inputs1.to(device)
    
            # trainloader2から拡張データ取得
            inputs2, labels2 = data2[0], data2[1].to(device)
            inputs2 = STFT(inputs2)
            inputs2 = torch.from_numpy(inputs2.astype(np.float32))
            inputs2 = inputs2.to(device)

            # 順伝播
            outputs1, outputs_sub1 = net1(inputs1)
            outputs2, outputs_sub2 = net1(inputs2)

            # 損失計算
            loss1 = criterion(outputs1, labels1) 
            loss2 = criterion(outputs2, labels2) 
            loss3 = mse_loss(outputs_sub1, outputs_sub2)
            loss = loss1 + loss2 + loss3

            # 誤差を逆伝播
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()

       ######################## test data ###############################
        #####################
        predicted_classes = [] 
        true_classes = []
        #####################
        with torch.no_grad():
            for data in testloader:
                inputs, labels = data[0], data[1].to(device)
                inputs=STFT(inputs)
                inputs = torch.from_numpy(inputs.astype(np.float32))
                inputs=inputs.to(device)
                #inputs=inputs.float()
                outputs,_ = net1(inputs)
                _, predictions = torch.max(outputs, 1)

                ##################################
                predicted_classes.extend(predictions.cpu().numpy())
                true_classes.extend(labels.cpu().numpy())
                ##################################]

        
        for plus_class in range(len(predicted_classes)):
            count_matrix[epoch][true_classes[plus_class]][predicted_classes[plus_class]] += 1

    print(count_matrix[EPOCH-1])

for i in range(EPOCH):
    for j in range(len(classes)):
        acc[i][j]+=count_matrix[i][j][j]
    if i==EPOCH-1:
        print(count_matrix[i])
        np.save('SinM8S', count_matrix[i])
