'''
NN_decoder学習用コード
データセットは学習時に自動生成
~/NN_decoderで実行する
'''

'''
< 次の実装 >
現在単一のエラーレートのみから訓練データをサンプリングしている
→複数のエラーレートから均一にサンプリングし、訓練データとする実装
'''

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time


from param import param
from toric_code import ToricCode

SIZE = param.code_distance
num_train_data = int(1e6)
num_test_data = int(1e5)
batch_size = int(1000)
EPOCH = 500
learning_rate = 1e-4

def generate_dataset(num_data, toric_code):
    data_list = []
    label_list = []
    for i in range(num_data):
        errors = toric_code.generate_errors()
        syn_x = torch.from_numpy(toric_code.generate_syndrome_X(errors))
        syn_z = torch.from_numpy(toric_code.generate_syndrome_Z(errors))
        syn = torch.cat((syn_x, syn_z), dim=0)
        errors_x, errors_z = toric_code.errors_to_errorsXZ(errors)
        errors_tensor_x = torch.from_numpy(errors_x)
        errors_tensor_z = torch.from_numpy(errors_z)
        errors_tensor = torch.cat((errors_tensor_x, errors_tensor_z), dim=0)
        data_list.append(syn.flatten())
        label_list.append(errors_tensor.flatten())
    return data_list, label_list

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        #self.flatten = nn.Flatten()
        self.linear_reru_stack = nn.Sequential(
            nn.Linear(2*SIZE**2, 8*SIZE**2),
            nn.Tanh(),
            nn.Linear(8*SIZE**2, 8*SIZE**2),
            nn.Tanh(),
            nn.Linear(8*SIZE**2, 8*SIZE**2),
            nn.Tanh(),
            nn.Linear(8*SIZE**2, 8*SIZE**2),
            nn.Tanh(),
            nn.Linear(8*SIZE**2, 8*SIZE**2),
            nn.Tanh(),
            nn.Linear(8*SIZE**2, 8*SIZE**2),
            nn.Tanh(),
            nn.Linear(8*SIZE**2, 8*SIZE**2),
            nn.Tanh(),
            nn.Linear(8*SIZE**2, 8*SIZE**2),
            nn.Tanh(),
            nn.Linear(8*SIZE**2, 8*SIZE**2),
            nn.Tanh(),
            nn.Linear(8*SIZE**2, 8*SIZE**2),
            nn.Tanh(),
            nn.Linear(8*SIZE**2, 8*SIZE**2),
            nn.Tanh(),
            nn.Linear(8*SIZE**2, 8*SIZE**2),
            nn.Tanh(),
            nn.Linear(8*SIZE**2, 8*SIZE**2),
            nn.Tanh(),
            nn.Linear(8*SIZE**2, 8*SIZE**2),
            nn.Tanh(),
            nn.Linear(8*SIZE**2, 4*SIZE**2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        logits = self.linear_reru_stack(x)
        return logits

class LabeledCustomDataset(Dataset):
    def __init__(self, data_tensor_list, target_tensor_list):
        self.data = data_tensor_list
        self.targets = target_tensor_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_sample = self.data[index]
        target_sample = self.targets[index]
        # ここで必要に応じて前処理を行う
        return data_sample, target_sample

if __name__ == '__main__':
    #デバイスの設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('device is ' + str(device))
    #データセットの生成
    toric_code = ToricCode()
    now = datetime.now()
    now_f = now.strftime("%Y-%m-%d %H:%M:%S")
    print(now_f)
    train_data_list, train_label_list = generate_dataset(num_data=num_train_data, toric_code=toric_code)
    train_dataset = LabeledCustomDataset(train_data_list, train_label_list)
    train_batch = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_list, test_label_list = generate_dataset(num_data=num_test_data, toric_code=toric_code)
    test_dataset = LabeledCustomDataset(test_data_list, test_label_list)
    test_batch = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    now = datetime.now()
    now_f = now.strftime("%Y-%m-%d %H:%M:%S")
    print(now_f)
    print('--------------------------------------------------------')
    print('generating data is completed')
    print('--------------------------------------------------------')
    #モデル、損失関数、最適化関数の定義
    model = NeuralNetwork().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adadelta(model.parameters())
    #各値を格納するリストの生成
    train_loss_list = []
    test_loss_list = []
    #エポックの実行
    now = datetime.now()
    now_f = now.strftime("%Y-%m-%d %H:%M:%S")
    print(now_f)
    print('--------------------------------------------------------')
    print('start training')
    print('--------------------------------------------------------')
    epoch = EPOCH
    for i in range(epoch):
        before = time.perf_counter()
        print('-----------------------')
        print("Epoch: {}/{}".format(i+1, epoch))
        train_loss = 0
        test_loss = 0

        model.train()
        for batch in train_batch:
            '''print(batch)'''
            x = batch[0].to(device)
            x = x.float()
            y = batch[1].to(device)
            y = y.float()
            '''print(x)
            print(y)'''
            pred = model(x)
            '''print(pred)'''
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        epoch_train_loss = train_loss / len(train_batch)

        model.eval()
        with torch.no_grad():
            for batch in test_batch:
                x = batch[0].to(device)
                x = x.float()
                y = batch[1].to(device)
                y = y.float()
                pred = model(x)
                loss = criterion(pred, y)
                test_loss += loss.item()
        epoch_test_loss = test_loss / len(test_batch)

        #訓練用データセットの取り替え
        if i % 1 == 0:
            num_replace = int(0.50 * num_train_data)
            new_data_list, new_label_list = generate_dataset(num_data=num_replace, toric_code=toric_code)
            train_data_list[:num_replace] = new_data_list
            train_label_list[:num_replace] = new_label_list
            train_dataset = LabeledCustomDataset(train_data_list, train_label_list)
            train_batch = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        print("Train_Loss     : {:.4f}".format(epoch_train_loss))
        print("Test_Loss      : {:.4f}".format(epoch_test_loss))
        spent = time.perf_counter() - before
        formatted_time = "{:.3f}".format(spent)
        print(f"time             : {formatted_time} seconds")

        train_loss_list.append(epoch_train_loss)
        test_loss_list.append(epoch_test_loss)
        
    print('--------------------------------------------------------')
    print('complete training')
    print('--------------------------------------------------------')
    now = datetime.now()
    now_f = now.strftime("%Y-%m-%d %H:%M:%S")
    print(now_f)
    #学習したモデルを保存
    current_directory = os.getcwd()
    model_directory_path = os.path.join(current_directory, 'learned_model')
    os.chdir(model_directory_path)
    torch.save(model.state_dict(), 'NN_' + str(SIZE) + '.pt')

    #学習の結果の可視化
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    png_directory_path = os.path.join(parent_directory, 'png')
    os.chdir(png_directory_path)
    x = list(range(epoch))
    y_train = train_loss_list
    y_test = test_loss_list
    plt.figure()
    plt.title('train and test loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(x, y_train, label='train')
    plt.plot(x, y_test, label='test')
    plt.legend()
    plt.savefig('loss.png')