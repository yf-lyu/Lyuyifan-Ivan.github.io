# RNN 循环神经网络

循环神经网络的输出取决于当下的输入和前一时刻的隐变量。应用到语言模型时，循环神经网络根据当前的输入预测下一时刻的输出值

## 模型结构

<div align=center>
<img src="./深度学习/02RNN/images/fig1.jpg"/>
</div>

- 更新隐藏状态：$h_t=\phi (W_{hh}h_{t-1}+W_{hx}x_{t-1}+b_h)$
- 输出：$o_t=W_{ho}h_t+b_o$

## 困惑度(perplexity)
如果想要压缩文本，我们可以根据当前词元集预测下一个词元。一个好的语言模型应该能让我们更准确地预测下一个词元。因此，它应该允许我们在压缩序列时花费更少的比特
- 衡量一个语言模型的好坏可以用平均交叉熵（通过一个序列中所有的个词元的交叉熵损失的平均值来衡量）：
$$\pi=\frac{1}{n}\sum_{i=1}^{n}-logp(x_t|x_{t-1},...)$$
p是语言模型的预测概率，$x_t$是真实词，这使得不同长度的文档的性能具有了可比性
- 历史原因，NLP使用困惑度$exp(\pi)$来衡量
$$exp(\frac{1}{n}\sum_{i=1}^{n}-logp(x_t|x_{t-1},...))$$

困惑度的最好的理解是“下一个词元的实际选择数的调和平均数”。
- 在最好的情况下，模型总是完美地估计标签词元的概率为1。 在这种情况下，模型的困惑度为1
- 在最坏的情况下，模型总是预测标签词元的概率为0。 在这种情况下，困惑度是正无穷大
- 在基线上，该模型的预测是词表的所有可用词元上的均匀分布。 在这种情况下，困惑度等于词表中唯一词元的数量

## 梯度裁剪
- 迭代中计算T个时间步上的梯度，在反向传播过程中产生长度为O(T)的矩阵乘法链，导致数值不稳定
- 梯度裁剪可以有效预防梯度爆炸
  - 如果梯度长度g超过$\theta$，那么投影回长度$\theta$，g表示所有层上的梯度放在一个向量
    $$g \Leftarrow min(1,\frac{\theta}{||g||})g$$
    梯度长度大于$\theta$时，$||g||=\frac{\theta}{||g||}||g||=\theta$

## 更多的应用RNNs

<div align=center>
<img src="./深度学习/02RNN/images/fig2.jpg"/>
</div>

## 手写复现RNN模型

### 独热编码
将每个索引映射为相互不同的单位向量：假设词表中不同词元的数目为$N（即len(vocab)）$，词元索引的范围为$0到N-1$。如果词元的索引是整数i，那么我们将创建一个长度为$N$的全0向量，并将第i处的元素设置为1。此向量是原始词元的一个独热向量。
```python
def create_onehot(labels, class_num):
  one_hot = torch.tensor(np.zeros((len(labels), class_num)), dtype=torch.int32)
  for index, label in enumerate(labels):
    one_hot[index, label] = 1
# 也可调用torch.nn.functional的API实现
torch.nn.functional.one_hot(torch.tensor([0, 2]), len(vocab))
```
我们每次采样的小批量数据形状是二维张量：（批量大小，时间步长）。one_hot函数将这样一个小批量数据转换成三维张量，张量的最后一个维度等于词表大小（len(vocab)）。

我们经常转换输入的维度，以便获得形状为（时间步长，批量大小，词表大小）的输出。这将使我们能够更方便地通过最外层的维度，一步一步地更新小批量数据的隐状态。
```python
X = torch.arange(10).reshape(2,5) # (批量大小, 时间步长)
X = torch.nn.functional.one_hot(X.T, len(vocab))
```

### 初始化模型参数

隐藏单元数num_hiddens是一个可调的超参数。当训练语言模型时，输入和输出来自相同的词表。因此，它们具有相同的维度，即词表的大小。

```python
def get_params(vocab_size, num_hiddens, device):
  num_inputs = num_outputs = vocab_size
  def normal(shape, device=device):
    return torch.randn(shape, device=device)*0.01
  Wxh = normal((num_inputs, num_hiddens))
  Whh = normal((num_hiddens, num_hiddens))
  bh = torch.zeros(num_hiddens, device=device)
  Who = normal((num_hiddens, num_outputs))
  bo = torch.zeros(num_outputs, device=device)
  params = [Wxh, Whh, bh, Who, bo]
  for param in params:
    param.requires_grad_(True)  # 参数支持求导
  return params

def init_hidden_state(batch_size, num_hiddens, device): # 初始化隐藏层状态
  return (torch.zeros((batch_size, num_hiddens), device=device),) # 可以初始化为全0，也可初始化随机数
```

### RNN模型结构
```python
class RNN_Model:
  def __init__(self, vocab_size, num_hiddens, batch_size, get_params, init_state, device):
    self.vocab_size = vocab_size    # 词向量大小
    self.num_hiddens = num_hiddens  # 隐藏层
    self.batch_size = batch_size    # 批量大小
    self.device = device
    self.init_state = init_state
    self.params = get_params(self.vocab_size, self.num_hiddens, self.device)
  def begin_init_state(self):
    return self.init_state(self.batch_size, self.num_hiddens, self.device)
  # RNN模型结构
  def forward(self, inputs, state):
    inputs = F.one_hot(inputs.T, self.vocab_size).type(torch.float32)
    Wxh, Whh, bh, Who, bo = self.params
    H, = state
    outputs = []
    for X in inputs:
      H = torch.tanh(torch.mm(X,Wxh) + torch.mm(H,Whh) + bh)  # 激活函数使用tanh()
      Y = torch.mm(H,Who) + bo
      outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
  def __call__(self, *args):
    return self.forward(*args)
```
输出形状是（时间步数$*$批量大小，词表大小），而隐状态形状保持不变，即（批量大小，隐藏单元数）。

```python
inputs = torch.arange(10).reshape((2, 5))
num_hiddens = 512
batch_size = inputs.shape[0]
device = d2l.try_gpu()
model = RNN_Model(len(vocab), num_hiddens, batch_size, get_params, init_hidden_state, device)
init_state = model.begin_init()
Y, new_state = model(inputs.to(device), init_state)
print(Y.shape, len(new_state), new_state[0].shape)
Output:
        torch.Size([10, 28]) 1 torch.Size([2, 512])
```