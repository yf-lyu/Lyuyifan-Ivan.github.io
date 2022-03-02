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

## Pandas JSON文件

JSON（JavaScript Object Notation，JavaScript 对象表示法），是存储和交换文本信息的语法，类似 XML

读取json文件内容

```python
import pandas as pd
df = pd.read_json('sites.json')
print(df.to_string())
```
- to_string() 用于返回 DataFrame 类型的数据，我们也可以直接处理 JSON 字符串

```python
import pandas as pd
data =[
    {
      "id": "A001",
      "name": "菜鸟教程",
      "url": "www.runoob.com",
      "likes": 61
    },
    {
      "id": "A002",
      "name": "Google",
      "url": "www.google.com",
      "likes": 124
    },
    {
      "id": "A003",
      "name": "淘宝",
      "url": "www.taobao.com",
      "likes": 45
    }
]
df = pd.DataFrame(data)
print(df)
Output:     id       name         url         likes
        0  A001    菜鸟教程  www.runoob.com     61
        1  A002     Google  www.google.com     124
        2  A003      淘宝  www.taobao.com       45
```

JSON 对象与 Python 字典具有相同的格式，所以我们可以直接将 Python 字典转化为 DataFrame 数据：
```python
import pandas as pd
# 字典格式的JSON
s = {
    "col1":{"row1":1,"row2":2,"row3":3},
    "col2":{"row1":"x","row2":"y","row3":"z"}
}
# 读取 JSON 转为 DataFrame
df = pd.DataFrame(s)
print(df)
Output:         col1 col2
        row1     1    x
        row2     2    y
        row3     3    z

```

## Pandas 数据清洗

数据清洗是对一些没有用的数据进行处理的过程

很多数据集存在数据缺失、数据格式错误、错误数据或重复数据的情况，如果要对使数据分析更加准确，就需要对这些没有用的数据进行处理

### Pandas 清洗空值

如果我们要删除包含空字段的行，可以使用dropna( )方法，语法格式如下：

```python
DataFrame.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
# axis：默认为 0，表示逢空值剔除整行，如果设置参数 axis＝1 表示逢空值去掉整列
# how：默认为 'any' 如果一行（或一列）里任何一个数据有出现 NA 就去掉整行，如果设置 how='all' 一行（或列）都是 NA 才去掉这整行
# thresh：设置需要多少非空值的数据才可以保留下来的
# subset：设置想要检查的列。如果是多个列，可以使用列名的 list 作为参数
# inplace：如果设置 True，将计算得到的值直接覆盖之前的值并返回 None，修改的是源数据
```

我们可以通过 isnull( ) 判断各个单元格是否为空

```python
import pandas as pd
df = pd.read_csv('property-data.csv')
print (df['NUM_BEDROOMS'])
print (df['NUM_BEDROOMS'].isnull())     # 对应元素不为空则True，否则填入False 空值(Nan)
```

接下来的实例演示了删除包含空数据的行

```python
# 移除 ST_NUM 列中字段值为空的行
import pandas as pd
df = pd.read_csv('property-data.csv')
df.dropna(subset=['ST_NUM'], inplace = True)
print(df.to_string())
```

我们也可以 fillna() 方法来替换一些空字段：

```python
import pandas as pd
df = pd.read_csv('property-data.csv')
df.fillna(12345, inplace = True)    # 使用 12345 替换空字段
print(df.to_string())
```

### Pandas 清洗格式错误数据

数据格式错误的单元格会使数据分析变得困难，甚至不可能

我们可以通过包含空单元格的行，或者将列中的所有单元格转换为相同格式的数据

### Pandas 清洗错误数据

数据错误也是很常见的情况，我们可以对错误的数据进行替换或移除

```python
# 将 age 大于 120 的设置为 120
import pandas as pd
person = {
  "name": ['Google', 'Runoob' , 'Taobao'],
  "age": [50, 200, 12345]    
}
df = pd.DataFrame(person)
for x in df.index:
  if df.loc[x, "age"] > 120:
    df.loc[x, "age"] = 120
print(df.to_string())
```
