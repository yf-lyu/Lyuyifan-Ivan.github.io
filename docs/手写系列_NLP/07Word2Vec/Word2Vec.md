# word2vec 词向量

## 什么是word2vec和Embeddings？
Word2Vec其实就是通过学习文本来用词向量的方式表征词的语义信息，即通过一个嵌入空间使得语义上相似的单词在该空间内距离很近

Embedding其实就是一个映射，将单词从原先所属的空间映射到新的多维空间中，也就是把原先词所在空间嵌入到一个新的空间中去

我们从直观角度上来理解一下，cat这个单词和kitten(小猫)属于语义上很相近的词，而dog和kitten则不是那么相近，iphone这个单词和kitten的语义就差的更远了。通过对词汇表中单词进行这种数值表示方式的学习（也就是将单词转换为词向量），能够让我们基于这样的数值进行向量化的操作从而得到一些有趣的结论。比如说，如果我们对词向量kitten、cat以及dog执行这样的操作：kitten - cat + dog，那么最终得到的嵌入向量（embedded vector）将与puppy(小狗)这个词向量十分相近

## 字向量 & 词向量

词向量的表达能力比字向量要强，且词向量理解语义信息更多，在样本足够多的情况下，用词向量的效果一般要好，如果样本量较小，用字向量反而效果会更好

字向量一般应用于古诗生成，古诗生成不太依赖于词，其主要是以字为单位生成

词向量一般应用于小说生成、文本分类、机器翻译

## word2vec模型
在word2vec模型中，主要有Skip-Gram和CBOW两种模型
- CBOW是给定上下文，来预测input word
- Skip-Gram是给定input word来预测上下文

<div align=center>
<img src="./image/20220306.png" width="500" height="300" />
</div>

## Skip-Gram模型
Skip-Gram实际上分为了两个部分，第一部分为建立模型，第二部分为通过模型获取嵌入词向量

### 模型细节
首先，神经网络只能接受数值输入，所以我们必须将单词进行one-hot编码，上面我们介绍的词向量发挥作用了。假设我们在训练数据中只能取出5000个不重复的单词作为词汇表，那么我们对每个单词编码都是的向量

模型的输入是10000为的向量，那么输出也是10000维（词汇表的大小）向量，它包含了10000个概率，每一个概率代表着当前词是输入样本中output word的概率大小

```python
import os, pickle
import numpy as np
import pandas as pd
import jieba
from tqdm import tqdm
# 停词操作，对一些"标点符号、啊、阿、哎、哎呀、哎哟、唉、俺"词过滤
def stop_word(file):
    with open(file, 'r', encoding='utf-8') as f:
        return f.read().split('\n')
# 分词，使用jieba工具分词
def cut_word(file):
    Stop_word = stop_word("stopwords.txt")
    ret_data = []
    all_data = pd.read_csv("数学原始数据.csv", encoding='gbk', names=['datas'])['datas']
    for line in all_data:
        ret_word = jieba.lcut(line)
        ret_data.append([word for word in ret_word if word not in Stop_word])
    return ret_data
# 获取三个主要参数word_2_index(词-索引)、index_2_word(索引-词)、word_one_hot(词-onehot)
def get_dict(all_word):
    index_2_word = []
    for line in all_word:
        for word in line:
            if word not in index_2_word:
                index_2_word.append(word)
    word_2_index = {word: index for index, word in enumerate(index_2_word)}
    word_size = len(word_2_index)
    word_one_hot = {}
    for word, index in word_2_index.items():    # 键值对
        one_hot = np.zeros((1, word_size))
        one_hot[0, index] = 1
        word_one_hot[word] = one_hot
    return word_2_index, index_2_word, word_one_hot
```

### 训练模型

假如我们有一个句子“The dog barked at the mailman”

- 首先我们选句子中间的一个词作为我们的输入词，例如我们选取“dog”作为input word
- 有了input word以后，我们再定义一个叫做<font color=red>skip_window</font>的参数，它代表着我们从当前input word的一侧（左边或右边）选取词的数量。如果我们设置skip_window=2，那么我们最终获得窗口中的词（包括input word在内）就是['The', 'dog'，'barked', 'at']。skip_window=2代表着选取左input word左侧2个词和右侧2个词进入我们的窗口，所以整个窗口大小span=2x2=4。另一个参数叫<font color=red>num_skips</font>，它代表着我们从整个窗口中选取多少个不同的词作为我们的output word，当skip_window=2，num_skips=2时，我们将会得到两组 (input word, output word) 形式的训练数据，即 ('dog', 'barked')，('dog', 'the')。
- 神经网络基于这些训练数据将会输出一个概率分布，这个概率代表着我们的词典中的每个词是output word的可能性。
  - 这句话有点绕，我们来看个栗子。第二步中我们在设置skip_window和num_skips=2的情况下获得了两组训练数据。假如我们先拿一组数据 ('dog', 'barked') 来训练神经网络，那么模型通过学习这个训练样本，会告诉我们词汇表中每个单词是“barked”的概率大小

模型的输出概率代表着到我们词典中每个词有多大可能性跟input word同时出现

我们选定句子“The quick brown fox jumps over lazy dog”，设定我们的窗口大小为2（skip_window=2），也就是说我们仅选输入词前后各两个词和输入词进行组合。下图中，蓝色代表input word，方框内代表位于窗口内的单词。

<div align=center>
<img src="./image/20220306_1.png" width="500" height="300" />
</div>

我们的模型将会从每对单词出现的次数中习得统计结果。例如，我们的神经网络可能会得到更多类似（“Soviet“，”Union“）这样的训练样本对，而对于（”Soviet“，”Sasquatch“）这样的组合却看到的很少。因此，当我们的模型完成训练后，给定一个单词”Soviet“作为输入，输出的结果中”Union“或者”Russia“要比”Sasquatch“被赋予更高的概率

```python
def softmax(x):
    exp_x = np.exp(x)
    sum_x = np.sum(exp_x, axis=1, keepdims=True)
    return exp_x/sum_x
if __name__ == "__main__":
    ret_data = cut_word("数学原始数据.csv")
    word_2_index, index_2_word, word_one_hot = get_dict(ret_data)
    word_size = len(word_2_index)
    embedding_num = 100
    lr = 0.01
    epoch = 20
    n_gram = 3
    w1 = np.random.normal(-1, 1, size=(word_size, embedding_num))
    w2 = np.random.normal(-1, 1, size=(embedding_num, word_size))
    for e in range(epoch):
        for line_word in tqdm(ret_data):
            for index, now_word in enumerate(line_word):
                other_words = line_word[max(index-n_gram, 0): index] + line_word[index+1: index+1+n_gram]
                now_word_onehot = word_one_hot[now_word]
                for other_word in other_words:
                    other_word_onehot = word_one_hot[other_word]
                    # 前向传播
                    hidden1 = now_word_onehot @ w1
                    hidden2 = hidden1 @ w2
                    pre = softmax(hidden2)
                    # 交叉熵求解loss，但此模型求loss意义不大
                    loss = -np.mean(other_word_onehot*np.log(pre))
                    # 反向传播
                    G = pre - other_word_onehot
                    delta_w2 = hidden1.T @ G
                    delta_hidden1 = G @ w2.T
                    delta_w1 = now_word_onehot.T @ delta_hidden1
                    # 更新梯度
                    w1 = w1 - delta_w1 * lr
                    w2 = w2 - delta_w2 * lr
    with open("word2vec.pkl", 'wb') as f:   # 模型参数保存
        pickle.dump([w1, word_2_index, index_2_word], f)
```


## word2vec中为什么使用负采样？
1.加速了模型计算
- 不同于原本每个训练样本更新所有的权重，负采样每次让一个训练样本仅仅更新一小部分的权重，这样就会降低梯度下降过程中的计算量

2.保证了模型训练的效果
- 其一模型每次只需要更新采样的词的权重，不用更新所有的权重，因为那样会很慢
- 其二中心词其实只跟它周围的词有关系，位置离着很远的词没有关系，也没必要同时训练更新

## 关于提升效果的技巧

1.增大训练样本，语料库越大，模型学习的可学习的信息会越多

2.增加window size，可以获得更多的上下文信息

3.增加embedding size可以减少信息的维度损失，但也不宜过大，我一般常用的规模为50-300