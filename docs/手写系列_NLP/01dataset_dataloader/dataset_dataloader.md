# Dataset_DataLoader
## 手写复现Dataset_DataLoader
```python
import numpy as np
class MyDataset:
    def __init__(self, all_data, batch_size, shuffle=True):
        self.all_data = all_data
        self.batch_size = batch_size
        self.shuffle = shuffle
    def __iter__(self):             # 当for中调用这个对象时，__iter__函数会被执行一次
        return DataLoader(self)     # self本身就是一个Dataset，self传进来后就拥有里面的所有数据了
    def __len__(self):
        return len(self.all_data)
class DataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.cursor = 0
        self.index = [i for i in range(len(self.dataset))]
        if self.dataset.shuffle:
            np.random.shuffle(self.index)
    def __next__(self):     # 需要加条件判断，避免出现无限循环
        if self.cursor >= len(self.dataset):    # 或者 len(self.dataset.all_data)
            raise StopIteration
        index = self.index[self.cursor: self.cursor+self.dataset.batch_size]
        batch_data = self.dataset.all_data[index]
        self.cursor += self.dataset.batch_size
        return batch_data
if __name__ == "__main__":
    all_data = np.array([1, 2, 3, 4, 5, 6, 7])
    batch_size = 2
    shuffle = True
    epoch = 2
    dataset = MyDataset(all_data, batch_size, shuffle)
    for i in range(epoch):
        for batch_data in dataset:  # 把一个对象放在for上时，第一次会自动调用这个对象的__iter__
            print(batch_data)
```

## Pytorch的Dataset和DataLoader
Dataset 是 PyTorch 中用来表示数据集的一个抽象类，我们的数据集可以用这个类来表示，至少需要覆写下面两个方法：

- \__len__：一般用来返回数据集大小
- \__getitem__：实现这个方法后，可以通过下标的方式 dataset[i] 的来取得第 i 个数据

DataLoader 本质上就是一个 iterable（内部定义了 \__iter__\() 方法），\__iter__\() 被定义成生成器，使用 yield 来返回数据，并利用多进程来加速 batch data 的处理，DataLoader 组装好数据后返回的是 Tensor 类型的数据

注意：DataLoader 是间接通过 Dataset 来获得数据的，然后进行组装成一个 batch 返回，因为采用了生成器，所以每次只会组装
一个 batch 返回，不会一次性组装好全部的 batch，所以 DataLoader 节省的是 batch 的内存，并不是指数据集的内存，数据集可以一开始就全部加载到内存里，也可以分批加载，这取决于 Dataset 中\__init__ 函数的实现

```python
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        # 因为数据集比较小，所以全部加载到内存里了
        data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[:,:-1])
        self.y_data = torch.from_numpy(data[:,[-1]])    # 最后一列为标签
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        return self.len
dataset = DiabetesDataset('diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset,   # 传递数据集
                          batch_size=32,     # 小批量的数据大小，每次加载一batch数据
                          shuffle=True,      # 打乱数据之间的顺序
                          num_workers=2)     # 使用多少个子进程来加载数据，默认为0, 代表使用主线程加载batch数据
for epoch in range(100):  # 训练 100 轮
    for i, data in enumerate(train_loader, 0):  # 每次惰性返回一个 batch 数据
        iuputs, label = data
```
