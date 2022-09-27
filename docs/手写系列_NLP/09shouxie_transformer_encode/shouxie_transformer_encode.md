# 手写Transformer编码器

## 整体框架

<img src="image/tf-整体框架.jpg" alt="tf-整体框架" style="zoom:50%;" />

## Encode细节图

<img src="image/encoder-详细图.png" alt="encoder-详细图" style="zoom: 67%;" />

主要包含有自注意力模块、残差网络、归一化、前馈传播模块。其中编码器包括两个子层：Self-Attention、Feed Forward

每一个子层的传输过程中都会有一个（残差网络+归一化）

## 手写源码

```python
import math
import torch
import collections
import numpy as np
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
from torch.autograd import Variable
```

归一化源码

```python
class LayerNorm(nn.Module):

  """
  构建一个LayerNorm Module
  LayerNorm的作用：对x归一化，使x的均值为0，方差为1
  LayerNorm计算公式：x-mean(x)/\sqrt{var(x)+\epsilon} = x-mean(x)/std(x)+\epsilon
  """
  def __init__(self, x_size, eps=1e-6) -> None:
​    super(LayerNorm, self).__init__()
​    self.ones_tensor = nn.Parameter(torch.ones(x_size))
​    self.zeros_tensor = nn.Parameter(torch.zeros(x_size))
​    self.eps = eps
  def forward(self, x):
​    mean = x.mean(-1, keepdim=True)
​    std = x.std(-1, keepdim=True)
​    return self.ones_tensor * (x - mean) / (std + self.eps) + self.zeros_tensor
```

自注意力机制，cross_attention则需要把query和key、value输入信息分开，key和value输入相同

```python
def self_attention(query, key, value, dropout=None, mask=None, return_score=True):
  d_k = query.size(-1)
  score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
  if mask is not None:
​    mask.cuda()
​    score = score.masked_fill(mask==0, -1e9)
  self_atten_softmax = F.softmax(score, dim=-1)
  if dropout is not None:
​    self_atten_softmax = dropout(self_atten_softmax)
  if return_score:
​    return torch.matmul(self_atten_softmax, value), self_atten_softmax
  return torch.matmul(self_atten_softmax, value)
```

多头注意力机制

```python
class MultiHeadAttention(nn.Module):
  def __init__(self, head, d_model, dropout=0.1) -> None:
​    super(MultiHeadAttention, self).__init__()
​    assert (d_model % head == 0)
​    self.d_k = d_model // head
​    self.head = head
​    self.d_model =d_model
​    self.linear_query = nn.Linear(d_model, d_model)
​    self.linear_key = nn.Linear(d_model, d_model)
​    self.linear_value = nn.Linear(d_model, d_model)
​    self.linear_out = nn.Linear(d_model, d_model)
​    self.dropout = nn.Dropout(dropout)
​    self.attn_softmax = None
  def forward(self, query, key, value, mask=None):
​    if mask is not None:
​      mask = mask.unsqueeze(1)
​    n_batch = query.size(0)
​    # [batch, seq, d_model] -> [batch, head, seq, d_k]
​    query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)
​    key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)
​    value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)
​    x, self.attn_softmax = self_attention(query, key, value, dropout=self.dropout, mask=mask)
​    x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)
​    return self.linear_out(x)
```

前馈传播

```python
class FeedForward(nn.Module):
  def __init__(self, d_model: int, d_ff: int, dropout=0.1) -> None:
​    """
​    :param d_model: FFN第一层输入的维度
​    :param d_ff: FNN第二层隐藏层输入的维度
​    :param dropout: drop比率
​    """
​    super(FeedForward, self).__init__()
​    self.w_1 = nn.Linear(d_model, d_ff)
​    self.w_2 = nn.Linear(d_ff, d_model)
​    self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
​    self.dropout_1 = nn.Dropout(dropout)
​    self.relu = nn.ReLU()
​    self.dropout_2 = nn.Dropout(dropout)
  def forward(self, x, res_net=False):
​    # x:[batch, seq_len, model_dim]
​    inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
​    if res_net: # 残差网络
​      return self.dropout_2(self.w_2(inter)) + x
​    return self.dropout_2(self.w_2(inter))
```



```python
class SublayerConnecttion(nn.Module):
  def __init__(self, d_model, dropout=0.1) -> None:
​    super(SublayerConnecttion, self).__init__()
​    self.layer_norm = LayerNorm(d_model)
​    self.dropout = nn.Dropout(p=dropout)
  def forward(self, x, sublayer):
​    return self.dropout(self.layer_norm(x + sublayer(x)))

def clone_module_to_modulelist(module, module_num):
  return nn.ModuleList([deepcopy(module) for _ in range(module_num)])
```

编码器层源码实现

```python
class EncoderLayer(nn.Module):
  def __init__(self, d_model, attn, feed_forward, dropout=0.1) -> None:
​    super(EncoderLayer, self).__init__()
​    self.attn = attn
​    self.feed_forward = feed_forward
​    self.sublayer_connection_list = clone_module_to_modulelist(SublayerConnecttion(d_model=d_model, dropout=dropout), 2)
  def forward(self, text, img):
​    x = self.sublayer_connection_list[0](text, lambda text:self.attn(text, img, img))
​    return self.sublayer_connection_list[1](x, self.feed_forward)
```



```python
class Encoder(nn.Module):
  def __init__(self, n_layer, encoder_layer) -> None:
​    super(Encoder, self).__init__()
​    self.encoder_layer_list = clone_module_to_modulelist(encoder_layer, n_layer)
  def forward(self, x, mask):
​    for encoder_layer in self.encoder_layer_list:
​      x = encoder_layer(x, mask)
​    return x
```

程序测试

```python
if __name__ == "__main__":
  bsz = 4
  x = torch.rand(size=(bsz, 80, 768))
  img = torch.rand(size=(bsz, 1, 768))
  attn = MultiHeadAttention(12, 768)
  feed_forward=FeedForward(768,768)
  Cross_atten = EncoderLayer(d_model=768, attn=deepcopy(attn), feed_forward=deepcopy(feed_forward))
  atten_score = Cross_atten(x, img)
  print(atten_score.shape)
```

