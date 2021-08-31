---
layout: mypost
title: 深度学习之图像分类（八）-- Batch Normalization
categories: [深度学习, 图像分类]
---

## 深度学习之图像分类（八）Batch Normalization

本节学习 Batch Normalization，学习视频源于 [Bilibili](https://www.bilibili.com/video/BV1T7411T7wa)，此章节参考博客 [Batch Normalization详解以及pytorch实验](https://blog.csdn.net/qq_37541097/article/details/104434557)，以及 [知乎回答](https://www.zhihu.com/question/38102762/answer/607815171)。

![img16](resnet-16.png)



### 1. 前言

Batch Normalization 是 Google 团队在 2015 年提出的，原始论文为 [Batch normalization: Accelerating deep network training by reducing internal covariate shift](http://proceedings.mlr.press/v37/ioffe15.pdf)。在深度学习中，因为网络的层数非常多，如果数据分布在某一层开始有明显的偏移，随着网络的加深这一问题会加剧 (这在 BN 的文章中被称之为 internal covariate shift)，进而导致模型优化的难度增加，甚至不能优化。所以，归一化就是要减缓这个问题。BN 目的是使得我们的一批（同一个 Batch）的 feature map 满足均值为 0，方差为 1 的分布。通过该方法能够加速网络的收敛并提高准确率。

![img14](resnet-14.png)



### 2. BN 原理

我们在图像预处理过程中通常会对图像进行标准化处理，这样能够加速网络的收敛，如下图所示，对于Conv1来说输入的就是满足某一分布的特征矩阵，但对于 Conv2 而言输入的 feature map 就不一定满足某一分布规律了（注意这里所说满足某一分布规律并不是指某一个 Feature Map 的数据要满足分布规律，理论上是指整个训练样本集所对应 Feature Map 的数据要满足分布规律）。Batch Normalization 的目的就是使我们的 Feature Map 满足均值为 0，方差为 1 的分布规律。

![img9](resnet-9.png)

原文中提到：“对于一个拥有 $d$ 维的输入 $x$ ，我们将对它的每一个维度进行标准化处理。”  假设我们输入的 $x$ 是 RGB 三通道的彩色图像，那么这里的 $d$ 就是输入图像的 channels 即 $d=3$，$x = (x^{(1)}, x^{(2)}, x^{(3)})$，其中 $x^{(1)}$ 就代表我们的 R 通道所对应的特征矩阵，依此类推。标准化处理也就是分别对我们的 R 通道，G 通道，B 通道进行处理，即 $\hat{x}^{(k)} = (x^{(k)} - \text{E}[x^{(k)}]) / \sqrt{\text{Var}[x^{(k)}]}$。原文提供了更加详细的计算公式：

![img10](resnet-10.png)

让 Feature Map 满足某一分布规律，理论上是指整个训练样本集所对应 Feature Map 的数据要满足分布规律，也就是说要计算出整个训练集的 Feature Map 的均值和方差，然后再进行标准化处理，对于一个大型的数据集明显是不可能的，所以论文中说的是 Batch Normalization，也就是我们计算一个 Batch 数据的 Feature Map ，然后进行标准化（Batch 越大越接近整个数据集的分布，效果越好）。

根据上图的公式我们知道 $\mu_B$ 代表着我们计算的 Feature Map 每个通道（Channel）的均值，注意 $\mu_B$ 是一个向量不是一个值，向量的每一个元素代表着一个通道（Channel）的均值。$\delta^2_B$ 代表着我们计算的 Feature Map 每个通道（Channel）的方差，注意 $\delta^2_B$ 是一个向量不是一个值，向量的每一个元素代表着一个通道（Channel）的方差，然后根据 $\mu_B$ 和 $\delta^2_B$ 计算标准化处理后得到的值。下图给出了一个计算均值和方差的示例：

![img11](resnet-11.png)

上图展示了一个 batch size 为 2（两张图片）的 Batch Normalization 的计算过程，假设 feature1、feature2 分别是由 image1、image2 经过一系列卷积池化后得到的特征矩阵，feature 的 channel 为 2，那么 $x^{(1)}$ 代表该 batch 的所有 feature 的 channel1 的数据，同理 $x^{(2)}$ 代表该 batch 的所有 feature 的 channel2 的数据。然后分别计算和的均值与方差，得到我们的 $\mu_B$ 和 $\delta^2_B$ 两个向量。然后在根据标准差计算公式分别计算每个channel 的特征值（公式中的 $\epsilon$ 是一个很小的常量，防止分母为零的情况）。

在我们训练网络的过程中，我们是通过一个 batch 一个 batch 的数据进行训练的，但是我们在预测过程中通常都是输入一张图片进行预测，此时 batch size 为1，如果在通过上述方法计算均值和方差就没有意义了。所以我们在训练过程中要去不断的计算每个 batch 的均值和方差，并使用移动平均 (moving average) 的方法记录统计的均值和方差，在我们训练完后我们可以近似认为我们所统计的均值和方差就等于我们整个训练集的均值和方差。然后在我们验证以及预测过程中，就使用我们统计得到的均值和方差进行标准化处理。

在原论文公式中不是还有 $\gamma,\beta$ 两个参数吗？是的，$\gamma$ 是用来调整数值分布的方差大小，$\beta$ 用来调节数值均值的位置。这两个参数是在反向传播过程中学习得到的，$\gamma$ 的默认值是1，$\beta$ 的默认值是0。
$$
y=\frac{\gamma}{\sqrt{\operatorname{Var}[x]+\epsilon}} \cdot x+\left(\beta-\frac{\gamma \mathrm{E}[x]}{\sqrt{\operatorname{Var}[x]+\epsilon}}\right)
$$



### 3. BN 实际使用

光看理论还不够，具体实践出真知！

在**训练过程**中，均值 $\mu_B$ 和方差 $\delta_B^2$ 是通过计算当前批次数据得到的，记为 $\mu_{now}$ 和 $\delta_{now}^2$， 而我们的**验证以及预测过程**中所使用的均值方差是一个统计量，记为 $\mu_{statistic}$ 和 $\delta_{statistic}^2$，其具体更新策略如下，其中 momentum默认取0.1：


$$
\begin{aligned}
\mu_{\text {statistic }+1} &=(1-\text { momentum }) * \mu_{\text {statistic }}+\text { momentum } * \mu_{\text {now }} \\
\sigma_{\text {statistic }+1}^{2} &=(1-\text { momentum }) * \sigma_{\text {statistic }}^{2}+\text { momentum } * \sigma_{\text {now }}^{2}
\end{aligned}
$$



**注意实现细节**，在pytorch中对当前批次 Feature Map 进行 BN 处理时所使用的 $\delta_{now}^2$ 是**总体标准差**，计算公式如下：


$$
\sigma_{\text {now }}^{2}=\frac{1}{m} \sum_{i=1}^{m}\left(x_{i}-\mu_{\text {now }}\right)^{2}
$$
在更新统计量 $\delta_{statistic}^2$，采用的 $\delta_{now}^2$ 是**样本标准差**，计算公式如下：


$$
\sigma_{\text {now }}^{2}=\frac{1}{m-1} \sum_{i=1}^{m}\left(x_{i}-\mu_{\text {now }}\right)^{2}
$$



pytorch 测试代码如下：

（1）bn\_process 函数是自定义的 bn 处理方法，验证是否和使用官方 bn 处理方法结果一致。在 bn\_process 中计算输入 batch 数据的每个通道（channel）的均值和标准差（标准差等于方差开平方），然后通过计算得到的均值和总体标准差对 Feature 每个维度进行标准化，然后使用均值和样本标准差更新统计均值和标准差。

（2）初始化统计均值是一个元素为 0 的向量，元素个数等于 channel 深度；初始化统计方差是一个元素为 1 的向量，元素个数等于 channel 深度，初始化 $\gamma = 1, \beta = 0$ 。

```python
import numpy as np
import torch.nn as nn
import torch
 
 
def bn_process(feature, mean, var):
    feature_shape = feature.shape
    for i in range(feature_shape[1]):
        # [batch, channel, height, width]
        feature_t = feature[:, i, :, :]
        mean_t = feature_t.mean()
        # 总体标准差
        std_t1 = feature_t.std()
        # 样本标准差
        std_t2 = feature_t.std(ddof=1)
 
        # bn process
        # 这里记得加上eps和pytorch保持一致
        feature[:, i, :, :] = (feature[:, i, :, :] - mean_t) / np.sqrt(std_t1 ** 2 + 1e-5)
        # update calculating mean and var
        mean[i] = mean[i] * 0.9 + mean_t * 0.1
        var[i] = var[i] * 0.9 + (std_t2 ** 2) * 0.1
    print(feature)
 
 
# 随机生成一个batch为2，channel为2，height=width=2的特征向量
# [batch, channel, height, width]
feature1 = torch.randn(2, 2, 2, 2)
# 初始化统计均值和方差
calculate_mean = [0.0, 0.0]
calculate_var = [1.0, 1.0]
# print(feature1.numpy())
 
# 注意要使用copy()深拷贝
bn_process(feature1.numpy().copy(), calculate_mean, calculate_var)
 
bn = nn.BatchNorm2d(2, eps=1e-5)
output = bn(feature1)
print(output)
```

打印出通过自定义 bn\_process 函数得到的输出以及使用官方 bn 处理得到输出，明显结果是一样的（只是精度不同）

![img12](resnet-12.png)



### 4. BN 的变种

本小节参考 [知乎回答](https://www.zhihu.com/question/38102762/answer/607815171)。

Batch Normalization 的思想非常简单，为深层网络的训练做出了很大贡献。因为有依赖于样本数目的缺陷，所以也被研究人员盯上进行改进。说的比较多的就是 **Layer Normalization** 与 **Instance Normalization**，**Group Normalization**。这些方法的差异主要在于计算 normalization 使用的元素集合不同。Batch Normalization 是 BxHxW，Layer Normalization 是CxHxW，Instance Normalization是 HxW，Group Normalization 是 GxHxW。

如果抛弃对 batch 的依赖，也就是每一个样本都单独进行 normalization，同时各个通道都要用到，就得到了**Layer Normalization**。跟 Batch Normalization 仅针对单个神经元不同，Layer Normalization 考虑了神经网络中一层的神经元。如果输出的 Feature Map 大小为 (N,C,H,W)，那么在每一层 Layer Normalization 就是基于 **CxHxW** 个数值进行求平均以及方差的操作。

**Layer Normalization** 把每一层的特征通道一起用于归一化，如果每一个特征层单独进行归一化呢？也就是限制在某一个特征通道内，那就是 **Instance normalization**。如果输出的 Feature Map 大小为 (N,C,H,W)，那么在每一层 Instance Normalization 就是基于 **HxW** 个数值进行求平均以及方差的操作。对于风格化类的图像应用，Instance Normalization通常能取得更好的结果，它的使用本来就是风格迁移应用中提出。

**Group Normalization** 是 Layer Normalization 和 Instance Normalization  的中间体， Group Normalization 将 channel 方向分 group，然后对每个 Group 内做归一化，算其均值与方差。如果输出的 Feature Map 大小为 (N,C,H,W)，将通道 C 分为 G 个组，那么 Group Normalization 就是基于**GxHxW** 个数值进行求平均以及方差的操作。我只想说，你们真会玩，要榨干所有可能性。

在 Batch Normalization 之外，有人提出了通用版本 **Generalized Batch Normalization**，有人提出了硬件更加友好的 **L1-Norm Batch Normalization** 等，不再一一讲述。另一方面，以上的 Batch Normalization，Layer Normalization，Instance Normalization 都是将规范化应用于输入数据 x，Weight normalization 则是对权重进行规范化。



### 5. 使用 BN 时的注意事项

* 训练时要将 training 参数设置为 True，在验证时将 training 参数设置为 False。在 pytorch 中可通过创建模型的 model.train() 和 model.eval() 方法控制。

* batch size 尽可能设置大点，设置小后表现可能很糟糕，设置的越大求的均值和方差越接近整个训练集的均值和方差。如果 batch 太小，则优先用 Group Normalization 替代。

* 建议将 bn 层放在卷积层（Conv）和激活层（例如Relu）之间，且卷积层不要使用偏置 bias，因为没有用，参考下图推理，即使使用了偏置 bias 求出的结果也是一样的 $y_i^b = y_i$
* 对于 RNN 等时序模型，有时候同一个 batch 内部的训练实例长度不一(不同长度的句子)，则不同的时态下需要保存不同的统计量，无法正确使用 BN 层，只能使用 Layer Normalization。
* 对于图像生成以及风格迁移类应用，使用 Instance Normalization 更加合适。

![img13](resnet-13.png)



### 6. 为什么 BN 能 Work？

Normalization 机制至今仍然是一个非常开放的问题，相关的理论研究一直都有。关于 Normalization 的有效性，有以下几个主要观点：

* **主流观点**，Batch Normalization 调整了数据的分布，不考虑激活函数，它让每一层的输出归一化到了均值为 0 方差为 1 的分布，这保证了梯度的有效性，目前大部分资料都这样解释，比如 BN 的原始论文[1] 认为的缓解了 Internal Covariate Shift(ICS) 问题。
* 可以使用更大的学习率，文[2] 指出 BN 有效是因为用上 BN 层之后可以使用更大的学习率，从而跳出不好的局部极值，增强泛化能力，在它们的研究中做了大量的实验来验证。
* 损失平面平滑。文[3] 的研究提出，BN 有效的根本原因不在于调整了分布，因为即使是在 BN 层后模拟ICS，也仍然可以取得好的结果。它们指出，BN 有效的根本原因是平滑了损失平面。零均值归一化/Z-score标准化 ($y_i = (x_i - \mu) / \sigma$) 对于包括孤立点的分布可以进行更平滑的调整。

> [1] Ioffe S, Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift[J]. arXiv preprint arXiv:1502.03167, 2015.
>
> [2] Bjorck N, Gomes C P, Selman B, et al. Understanding batch normalization[C]//Advances in Neural Information Processing Systems. 2018: 7705-7716.
>
> [3] Santurkar S, Tsipras D, Ilyas A, et al. How does batch normalization help optimization?[C]//Advances in Neural Information Processing Systems. 2018: 2488-2498.



附：李宏毅老师 Batch Normalization 的视频讲解 [https://www.bilibili.com/video/av9770302?p=10](https://www.bilibili.com/video/av9770302?p=10 ) 