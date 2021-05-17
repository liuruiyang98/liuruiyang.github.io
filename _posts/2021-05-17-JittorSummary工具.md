---
layout: mypost
title: JittorSummary工具
categories: [Jittor]
---

清华大学**计**算机系**图**形学实验室提出了一个全新的深度学习框架——**计图** (Jittor)，它是一个**采用**元算子表达神经网络计算单元**、**完全基于动态编译（Just-in-Time）的深度学习框架。详情可见 [Jittor官网](https://cg.cs.tsinghua.edu.cn/jittor/)。本博客介绍了从 torchsummary 迁移的 jittorsummary 工具的使用方法，源代码链接点击 [此处](https://github.com/liuruiyang98/Jittor-summary)。方便使用的话欢迎大家 Star!

## 1. 使用

- `git clone https://github.com/liuruiyang98/Jittor-summary.git`

```python
from jittorsummary import summary
summary(your_model, input_size=(channels, H, W), device='cpu')
```

- `input_size` 需要保证网络能够正确前向推理。
-  jittorsummary 支持 **cuda**。
   - `device = ‘cpu’`  ===> `jt.flags.use_cuda = 0`
   - `device = ‘cuda’`  ===> `jt.flags.use_cuda = 1`
- 编写 jittorsummary 时 jittor 版本号为 **1.2.2.34**. 出于部分 jittor 开发原因，中间某些版本不可用。之后 jittor version >= **1.2.2.60** 是可用的。


## 2. 样例

### 2.1 CNN for MNIST

```python
import jittor as jt
from jittor import init
from jittor import nn
from jittorsummary import summary

class SingleInputNet(nn.Module):

    def __init__(self):
        super(SingleInputNet, self).__init__()
        self.conv1 = nn.Conv(1, 10, 5)
        self.conv2 = nn.Conv(10, 20, 5)
        self.conv2_drop = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def execute(self, x):
        x = nn.relu(nn.max_pool2d(self.conv1(x), 2))
        x = nn.relu(nn.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(((-1), 320))
        x = nn.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.log_softmax(x, dim=1)
      
model = SingleInputNet()
summary(model, (1, 28, 28))
```

```txt
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
              Conv-1           [-1, 10, 24, 24]             260
              Conv-2             [-1, 20, 8, 8]           5,020
           Dropout-3             [-1, 20, 8, 8]               0
            Linear-4                   [-1, 50]          16,050
            Linear-5                   [-1, 10]             510
    SingleInputNet-6                   [-1, 10]               0
================================================================
Total params: 21,840
Trainable params: 21,840
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.06
Params size (MB): 0.08
Estimated Total Size (MB): 0.15
----------------------------------------------------------------
```



### 2.2 Multiple Inputs

```python
import jittor as jt
from jittor import init
from jittor import nn
from jittorsummary import summary

class MultipleInputNet(nn.Module):

    def __init__(self):
        super(MultipleInputNet, self).__init__()
        self.fc1a = nn.Linear(300, 50)
        self.fc1b = nn.Linear(50, 10)
        self.fc2a = nn.Linear(300, 50)
        self.fc2b = nn.Linear(50, 10)

    def execute(self, x1, x2):
        x1 = nn.relu(self.fc1a(x1))
        x1 = self.fc1b(x1)
        x2 = nn.relu(self.fc2a(x2))
        x2 = self.fc2b(x2)
        x = jt.contrib.concat((x1, x2), dim=0)
        return nn.log_softmax(x, dim=1)
      
model = MultipleInputNet()
summary(model, [(1, 300), (1, 300)])
```

```txt
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                [-1, 1, 50]          15,050
            Linear-2                [-1, 1, 10]             510
            Linear-3                [-1, 1, 50]          15,050
            Linear-4                [-1, 1, 10]             510
  MultipleInputNet-5                [-1, 1, 10]               0
================================================================
Total params: 31,120
Trainable params: 31,120
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.34
Forward/backward pass size (MB): 0.00
Params size (MB): 0.12
Estimated Total Size (MB): 0.46
----------------------------------------------------------------
```



### 2.3 Multiple Ouputs

```python
import jittor as jt
from jittor import init
from jittor import nn
from jittorsummary import summary

class MultipleOutputNet(nn.Module):
    def __init__(self):
        super(MultipleOutputNet, self).__init__()
        self.conv1 = nn.Conv(1, 10, 5)
        self.conv2 = nn.Conv(10, 20, 5)
        self.conv2_drop = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def execute(self, x):
        x = nn.relu(nn.max_pool2d(self.conv1(x), 2))
        x = nn.relu(nn.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(((- 1), 320))
        x = nn.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.log_softmax(x, dim=1), x
     
model = MultipleOutputNet()
summary(model, (1, 28, 28))
```

```txt
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
              Conv-1           [-1, 10, 24, 24]             260
              Conv-2             [-1, 20, 8, 8]           5,020
           Dropout-3             [-1, 20, 8, 8]               0
            Linear-4                   [-1, 50]          16,050
            Linear-5                   [-1, 10]             510
 MultipleOutputNet-6       [[-1, 10], [-1, 10]]               0
================================================================
Total params: 21,840
Trainable params: 21,840
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.06
Params size (MB): 0.08
Estimated Total Size (MB): 0.15
----------------------------------------------------------------
```



### 2.4 CUDA support

```python
import jittor as jt
from jittor import init
from jittor import nn
from jittorsummary import summary

class SingleInputNet(nn.Module):

    def __init__(self):
        super(SingleInputNet, self).__init__()
        self.conv1 = nn.Conv(1, 10, 5)
        self.conv2 = nn.Conv(10, 20, 5)
        self.conv2_drop = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def execute(self, x):
        x = nn.relu(nn.max_pool2d(self.conv1(x), 2))
        x = nn.relu(nn.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(((-1), 320))
        x = nn.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.log_softmax(x, dim=1)
      
model = SingleInputNet()
summary(model, (1, 28, 28), device='cuda')
```



### 2.5 Try more models

我提供了 **UNet, UNet++** 和 **Dense-UNet** 的 pytorch 以及 jittor 实现。方便比较  `torchsummary` 和 `jittorsummary` 的运行结果。

```txt
|- jittorsummary
	|- tests
		|- test_models
			|- DenseUNet_jittor.py
			|- DenseUNet_pytorch.py
			|- NestedUNet_jittor.py
			|- NestedUNet_pytorch.py
			|- UNet_jittor.py
			|- UNet_pytorch.py
```



## 3. Pytorch-to-Jittor

- 请参照 [pytorch-to-jittor](https://cg.cs.tsinghua.edu.cn/jittor/tutorial/2020-5-2-16-43-pytorchconvert/) 提供的文档进行 pytorch 模型到 jittor 模型的转换
- 在线转换工具：[pt-converter](https://cg.cs.tsinghua.edu.cn/jittor/pt_converter/).



## References

- The idea for this package sparked from [pytorch-summary](https://github.com/sksq96/pytorch-summary).