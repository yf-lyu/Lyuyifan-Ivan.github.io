# Python的numpy

## numpy数据类型

① 创建numpy数组的时候可以通过属性dtype显示指定数据的类型

② 在不指定数据类型的情况下，numpy会自动推断出适合的数据类型

③ 如果需要更改一个已经存在的数组的数据类型，可以通过astype方法进行修改从而得到一个新数组

```python
a2 = np.array([1,2,3,4])    # 自动推断出合适的数据类型，里面无浮点数，变为int32
print(a2.dtype)             # int32
a3 = a2.astype(float)       # astype得到的是一个新数组，原数组没有改变。
print(a2.dtype)             # int32
print(a2)                   # [1 2 3 4]
print(a3.dtype)             # float64
print(a3)                   # [1. 2. 3. 4.]
```

## numpy常用方法

① arange( )函数：类似python的range函数，通过指定开始值、终值和步长来创建一个一维数组，注意：最终创建的数组不包含终值

② linspace( )函数：通过指定开始值、终值和元素个数来创建一个一维数组，数组的数据元素符合等差数列，可以通过endpoint关键字指定是否包含终值，默认包含终值

③ logspace( )函数：和linspace函数类似，不过创建的是等比数列数组

④ random( )函数：创建0-1之间的随机元素，数组包含的元素数量由参数决定

⑤ zeros( )函数：创建值为0的数组

⑥ ones( )函数：创建值为1的数组

⑦ empty( )函数：创建元素值为随机数的数组

```python
import numpy as np
x = np.arange(10, 25, 5)                # 创建均匀间隔的数组（步进值为5）
x = np.linspace(0, 2, 9)                # 创建均匀间隔的数组（样本数为9）
x = np.random.random((3, 2))            # 创建随机值的数组
x = np.zeros((3, 2))                    # 创建值为0的数组
x = np.ones((3, 2))                     # 创建值为1的数组
x = np.empty((3, 2), dtype = np.int)    # 产生3行2列的二维数组，数组中每个元素都是随机数
```

## numpy输入/输出

```python
import numpy as np
all_data = np.loadtxt("my_file.txt")                    # 加载txt文本
np.savetxt("my_file.txt", all_data, delimiter = " ")    # 保存txt文本到指定路径
```

## numpy修改形状

① 对于一个已经存在的ndarray数组对象而言，可以通过修改形状相关的参数/方法从而改变数组的形状

② 直接使用reshape函数创建一个改变尺寸的新数组，原数组的shape保持不变，但是新数组和原数组共享一个内存空间，也就是修改任何一个数组中的值都会对另外一个产生影响，另外要求新数组的元素个数和原数组一致

```python
import numpy as np
x = np.empty((8, 1), dtype = np.int)
x = x.reshape(4, 2)                     # 改变数组形状，但不改变数据
x = x.ravel()                           # 拉平数组
```

## numpy转置

① 转置时重塑的一种特殊形式，它返回的是源数据的视图(不会进行任何赋值操作)

② 数组不仅有transpose方法，还有一个特殊的T属性

```python
import numpy as np
arr = np.arange(15).reshape((3,5))
# 方法一：
print(arr.T)               # 用数组的T方法进行转置
# 方法二：
print(np.transpose(arr))   # 用transpose方法一进行转置
# 方法三：
print(arr.transpose(1,0))  # 用transpose方法二进行转置
print(arr)                 # 源数据没有变化
```

## numpy数组合并
```python
arr = np.arange(6).reshape((2,3))
print(np.hstack((arr,arr)))                # 水平方向合并
print(np.vstack((arr,arr)))                # 垂直方向合并
print(np.concatenate((arr,arr),axis = 1))  # 指定对x轴进行拼接
print(np.concatenate((arr,arr),axis = 0))  # 指定对y轴进行拼接
```

## numpy聚合函数
① 聚合函数是对一组值(例如一个数组)进行操作，返回一个单一值作为结果的函数

② 聚合函数也可以指定对某个具体的轴进行数据聚合操作；常用的聚合操作有：平均值、最大值、最小值、方差等待

```python
import numpy as np
arr = np.array([[1,-2],[3,4]])
print(np.abs(arr))      # 取绝对值
print(np.sqrt(arr))     # 负数不能开根号
print(np.isnan(arr))    # 判断元素是否为nan值
print(arr.sum())        # 数组所有元素求和
print(arr.sum(axis=0))  # 将数组行求和，即求和每一列的值
print(arr.sum(axis=1))  # 将数组列求和，即求和每一行的值
print(arr.mean())       # 平均数
print(arr.max())        # 数组最大值    arr.max(axis=0)
print(np.std(arr))      # 标准差
```

## numpy数组复制、排序
```python
import numpy as np
arr = np.array([[1,-2],[3,4]])
x1 = np.copy(arr)               # 创建数组的副本
x2 = arr.copy()                 # 创建数组的深度拷贝
print(np.sort(arr, axis=0))     # 按列排序
```

## numpy数组切片、索引
```python
import numpy as np
arr1 = np.arange(1, 6, 2)
arr2 = np.array([[1, -2, 5],[3, 4, 2.6]])
print(arr1[0:2])        # [1 3] 选择索引为0与1对应的值
print(arr2[0:2, 1])     # [-2. 4.] 选择第1列中第0行、第1行的值
print(arr2[:1])         # [[1. -2. 5.]] 选择第0行的所有值，等同于b[0:1, :1]
# 条件索引
print(arr1[arr1<3])
# 花式索引
x=np.arange(32).reshape((8,4))
print(x[[4,2,1,7]])
output:[[16 17 18 19]
        [ 8  9 10 11]
        [ 4  5  6  7]
        [28 29 30 31]]
```
