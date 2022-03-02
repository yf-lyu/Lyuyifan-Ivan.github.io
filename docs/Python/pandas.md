# Python的pandas

① Pandas 一个强大的分析结构化数据的工具集，基础是 Numpy（提供高性能的矩阵运算）

② Pandas 可以从各种文件格式比如 CSV、JSON、SQL、Microsoft Excel 导入数据

③ Pandas 可以对各种数据进行运算操作，比如归并、再成形、选择，还有数据清洗和数据加工特征

## pandas 数据结构-Series

Pandas Series 类似表格中的一个列（column），类似于一维数组，可以保存任何数据类型

Series 由索引（index）和列组成

```python
pandas.Series(data, index, dtype, name, copy)
# data：一组数据(ndarray 类型)
# index：数据索引标签，如果不指定，默认从 0 开始
# dtype：数据类型，默认会自己判断
# name：设置名称
# copy：拷贝数据，默认为 False
```

创建一个简单的 Series 实例：

```python
import pandas as pd
a = [1, 2, 3]
myvar = pd.Series(a)
print(myvar)
Output: 0    1
        1    2
        2    3
        dtype: int64
```

我们可以指定索引值，如下实例：

```python
import pandas as pd
a = ["Google", "Runoob", "Wiki"]
myvar = pd.Series(a, index = ["x", "y", "z"])
print(myvar)
Output: x    Google
        y    Runoob
        z      Wiki
        dtype: object
```

我们也可以使用 key/value 对象，类似字典来创建 Series：

```python
import pandas as pd
sites = {1: "Google", 2: "Runoob", 3: "Wiki"}
myvar = pd.Series(sites)
print(myvar)
Output: 1    Google
        2    Runoob
        3      Wiki
        dtype: object
```

设置 Series 名称参数：

```python
import pandas as pd
sites = {1: "Google", 2: "Runoob", 3: "Wiki"}
myvar = pd.Series(sites, index = [1, 2], name="RUNOOB-Series-TEST" )
print(myvar)
Output: 1    Google
        2    Runoob
        Name: RUNOOB-Series-TEST, dtype: object
```

## Pandas 数据结构-DataFrame

DataFrame 是一个表格型的数据结构，它含有一组有序的列，每列可以是不同的值类型（数值、字符串、布尔型值）。DataFrame 既有行索引也有列索引，它可以被看做由 Series 组成的字典（共同用一个索引）

```python
pandas.DataFrame( data, index, columns, dtype, copy)
# data：一组数据(ndarray、series, map, lists, dict等类型)
# index：索引值，或者可以称为行标签
# columns：列标签，默认为 RangeIndex (0, 1, 2, …, n) 
# dtype：数据类型
# copy：拷贝数据，默认为 False
```

Pandas DataFrame 是一个二维的数组结构，类似二维数组

```python
import pandas as pd
data = [['Google',10],['Runoob',12],['Wiki',13]]
df = pd.DataFrame(data,columns=['Site','Age'],dtype=float)
print(df)
Output:     Site   Age
        0  Google  10.0
        1  Runoob  12.0
        2    Wiki  13.0
# 另一种写法
import pandas as pd
data = {'Site':['Google', 'Runoob', 'Wiki'], 'Age':[10, 12, 13]}
df = pd.DataFrame(data)
print (df)
```

还可以使用字典（key/value），其中字典的 key 为列名:

```python
import pandas as pd
data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
df = pd.DataFrame(data)
print (df)
Output:    a   b     c
        0  1   2   NaN
        1  5  10  20.0
```

Pandas 可以使用 loc 属性返回指定行的数据，如果没有设置索引，第一行索引为 0，第二行索引为 1，以此类推：

```python
import pandas as pd
data = {
        "calories": [420, 380, 390],
        "duration": [50, 40, 45]}
# 数据载入到 DataFrame 对象
df = pd.DataFrame(data)
# 返回第一行
print(df.loc[0])
# 返回第二行
print(df.loc[1])
# 返回第一行和第二行
print(df.loc[[0, 1]])
Output:        calories    420
                duration     50
                Name: 0, dtype: int64   # 返回结果其实就是一个Pandas Series数据
                calories    380
                duration     40
                Name: 1, dtype: int64   # 返回结果其实就是一个Pandas Series数据
                
                calories  duration      # 返回结果其实就是一个Pandas DataFrame数据
            0       420        50
            1       380        40
```

## Pandas CSV文件

CSV（Comma-Separated Values，逗号分隔值，有时也称为字符分隔值，因为分隔字符也可以不是逗号），其文件以纯文本形式存储表格数据（数字和文本）

```python
import pandas as pd
df = pd.read_csv('nba.csv')
print(df.to_string())
```
- to_string() 用于返回 DataFrame 类型的数据，如果不使用该函数，则输出结果为数据的前面 5 行和末尾 5 行，中间部分以 ... 代替

我们也可以使用 to_csv() 方法将 DataFrame 存储为 csv 文件：

```python
import pandas as pd
# 三个字段 name, site, age
nme = ["Google", "Runoob", "Taobao", "Wiki"]
st = ["www.google.com", "www.runoob.com", "www.taobao.com", "www.wikipedia.org"]
ag = [90, 40, 80, 98]
# 字典
dict = {'name': nme, 'site': st, 'age': ag}
df = pd.DataFrame(dict)
# 保存 dataframe
df.to_csv('site.csv')
```