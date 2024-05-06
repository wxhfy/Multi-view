import numpy as np
import scipy
from scipy.io import loadmat
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import torch
import h5py

# N 样本数量 K 类别数量 V 视图数量 n_input 输入维度 n_hid 隐藏层维度 n_output 输出维度
data_info = dict(
    Leaves={1: '100Leaves', 'N': 1600, 'K': 100, 'V': 3, 'n_input': [64, 64, 64], 'n_hid': [512, 512], 'n_output': 64},
    animal={1: 'animal', 'N': 10158, 'K': 50, 'V': 2, 'n_input': [4096, 4096], 'n_hid': [512, 512], 'n_output': 64},
    ALOI={1: 'ALOI-100', 'N': 10800, 'K': 100, 'V': 4, 'n_input': [77, 13, 64, 125], 'n_hid': [512, 512], 'n_output': 64},
    BBCSport={1: 'BBCSport', 'N': 544, 'K': 5, 'V': 2, 'n_input': [3183, 3203], 'n_hid': [512, 512], 'n_output': 64},
    BDGP={1: 'BDGP', 'N': 2500, 'K': 5, 'V': 2, 'n_input': [1750, 79], 'n_hid': [512, 512], 'n_output': 64},
    caltech7={1: 'caltech7', 'N': 1474, 'K': 7, 'V': 2, 'n_input': [512, 928], 'n_hid': [512, 512], 'n_output': 64},
    caltech20={1: 'caltech20', 'N': 2386, 'K': 20, 'V': 2, 'n_input': [1984, 512], 'n_hid': [512, 512], 'n_output': 64},
    caltech20_2={1: 'caltech20_2', 'N': 2386, 'K': 20, 'V': 6, 'n_input': [48, 40, 254, 1984, 512, 928], 'n_hid': [512, 512], 'n_output': 64},
    Caltech101={1: 'Caltech101-all_fea', 'N': 9144, 'K': 101, 'V': 2, 'n_input': [4096, 4096], 'n_hid': [512, 512], 'n_output': 64},
    Caltech={1: 'Caltech-V5', 'N': 1400, 'K': 7, 'V': 5, 'n_input': [40, 254, 1984, 512, 928], 'n_hid': [512, 512], 'n_output': 64},
    Cora={1: 'Cora', 'N': 2708, 'K': 7, 'V': 4, 'n_input': [2708, 1433, 2708, 2708], 'n_hid': [512, 512], 'n_output': 64},
    flowers17={1: 'flowers17', 'N': 1360, 'K': 17, 'V': 7, 'n_input': [1360, 1360, 1360, 1360, 1360, 1360, 1360], 'n_hid': [512, 512], 'n_output': 64},
    HW={1: 'HW', 'N': 10000, 'K': 10, 'V': 2, 'n_input': [784, 256], 'n_hid': [512, 512], 'n_output': 64},
    Mfeat={1: 'Mfeat', 'N': 10000, 'K': 10, 'V': 2, 'n_input': [784, 256], 'n_hid': [512, 512], 'n_output': 64},
    mfeat2={1: 'mfeat2', 'N': 2000, 'K': 10, 'V': 2, 'n_input': [76, 240], 'n_hid': [512, 512], 'n_output': 64},
    MNIST_USPS={1: 'MNIST_USPS', 'N': 5000, 'K': 10, 'V': 2, 'n_input': [784, 784], 'n_hid': [512, 512], 'n_output': 64},
    MSRC={1: 'MSRC', 'N': 210, 'K': 7, 'V': 5, 'n_input': [24, 576, 512, 256, 254], 'n_hid': [512, 512], 'n_output': 64},
    MSRC_v1={1: 'MSRC_v1', 'N': 210, 'K': 7, 'V': 5, 'n_input': [24, 576, 512, 256, 254], 'n_hid': [512, 512],
             'n_output': 64},
    ORL={1: 'ORL', 'N': 400, 'K': 4, 'V': 4, 'n_input': [512, 59, 864, 254], 'n_hid': [512, 512], 'n_output': 64},
    ORL_64x64={1: 'ORL_64x64', 'N': 400, 'K': 40, 'V': 1, 'n_input': [4096], 'n_hid': [512, 512], 'n_output': 64},
    Reuters={1: 'Reuters', 'N': 7200, 'K': 6, 'V': 5, 'n_input': [4819, 4810, 4892, 4858, 4777], 'n_hid': [512, 512], 'n_output': 64},
    Wiki={1: 'Wiki', 'N': 2866, 'K': 10, 'V': 2, 'n_input': [128, 10], 'n_hid': [512, 512], 'n_output': 64},
    UCI_Digits={1: 'UCI_Digits', 'N': 2000, 'K': 10, 'V': 6, 'n_input': [240, 76, 216, 47, 64, 6], 'n_hid': [512, 512],
                'n_output': 64},
    Yale={1: 'Yale', 'N': 165, 'K': 15, 'V': 3, 'n_input': [4096, 3304, 6750], 'n_hid': [512, 512], 'n_output': 64},
)


class GetData(Dataset):
    def __init__(self, name):
        data_path = './data_list/{}.mat'.format(name[1])
        np.random.seed(1)
        index = [i for i in range(name['N'])]
        np.random.shuffle(index)
        try:
            data = h5py.File(data_path)
        except OSError:
            data = loadmat(data_path)
        Final_data = []
        for i in range(name['V']):
            try:
                # 尝试第一种索引方式
                diff_view = data['data'][0][i]
            except:
                # 如果第一种索引方式失败，尝试第二种索引方式
                diff_view = data['data'][i][0]
            if isinstance(diff_view, scipy.sparse.csc.csc_matrix):
                diff_view = diff_view.toarray()  # 转换为密集矩阵
            try:
                diff_view = np.array(diff_view, dtype=np.float32)
            except:
                diff_view = np.array(diff_view, dtype=np.float32).T
            mm = MinMaxScaler()
            std_view = mm.fit_transform(diff_view)
            shuffle_diff_view = std_view[index]
            Final_data.append(shuffle_diff_view)
        if data['truelabel'].shape[0] == 1:
            label = np.array(data['truelabel'])
        else:
            label = np.array(data['truelabel']).T
        LABELS = label[index]
        self.name = name
        self.data = Final_data
        self.y = LABELS

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        data = [torch.from_numpy(self.data[i][idx]) for i in range(self.name['V'])]
        return data, torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()


def load_data(data_name):
    dataset_para = data_info[data_name]
    dataset = GetData(dataset_para)
    print("Dataset size:", len(dataset))
    dims = dataset_para['n_input']
    view = dataset_para['V']
    data_size = dataset_para['N']
    class_num = dataset_para['K']
    return dataset, dims, view, data_size, class_num

load_data('100Leaves')