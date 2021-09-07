---
layout: mypost
title: 深度学习之图像分类（十六）-- EfficientNetV2 网络结构
categories: [深度学习, 图像分类]
---

## 深度学习之图像分类（十六）EfficientNetV2 网络结构

本节学习 EfficientNetV2 网络结构。学习视频源于 [Bilibili](https://www.bilibili.com/video/BV19v41157AU)，博客参考 [EfficientNetV2网络详解](https://blog.csdn.net/qq_37541097/article/details/116933569)。

![img0](efficientnetv2-0.png)



### 1. 前言

EfficientNetV2 是 2021 年 4 月发表于 CVPR 的，其原始论文为 [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/pdf/2104.00298)。下图给出了EfficientNetV2 的性能，可其分为 S，M，L，XL 几个版本，在 ImageNet 21k 上进行预训练后，迁移参数到 ImageNet 1k 分类可见达到的正确率非常之高。相比而言 ViT 预训练后的性能也低了快两个点，训练速度也会更慢。

![img1](efficientnetv2-1.png)

![img2](efficientnetv2-2.png)

（仔细看上面两个图，ViT-L/16(21k) 在图中训练时间不到4天，表中要172小时，应该是表错了？）



在 EfficientNetV1 中作者关注的是准确率，参数数量以及 FLOPs（理论计算量小不代表推理速度快），在 EfficientNetV2 中作者进一步关注模型的**训练速度**。（其实我们更关心准确率和推理速度）。在表中可见，V2 相比 V1 在训练时间和推理时间上都有较大的优势。

![img3](efficientnetv2-3.png)



EfficientNetV2  值得关注的点在于两个方面：

* 采用新的网络模块：Fused-MBConv
* 采用渐进式学习策略，使得训练更快



### 2. 从 EfficientNetV1 到 EfficientNetV2

EfficientNetV1 存在的几个问题：

* 训练图像的尺寸很大时，训练速度非常慢
* 在网络浅层中使用 Depthwise convolutions 速度会很慢
* 同等的放大每个 stage 是次优的



`训练图像的尺寸很大时，训练速度非常慢`：表二中以 EfficientNet-B6 来进行实验的，当输入尺寸为 380 时，batch size 设置为 12 时，V100 每个 GPU 每秒能推理 37 张图片；batch size 设置为 24 时，每秒能推理 52 张。但是当输入尺寸为 512 时候，gpu 显存都会超，OOM(out of memory)。当网络中使用 BN 模块时，我们是希望 batch\_size 尽可能更大的。但是因为图像尺寸大，并不能把 batch 弄大。所以针对`训练图像的尺寸很大时，训练速度非常慢`问题，很自然想到降低训练图片尺寸，加快训练速度的同时还可以使用更大的 batch\_size。

![img4](efficientnetv2-4.png)



`在网络浅层中使用 Depthwise convolutions 速度会很慢`：原因是 DW 卷积在现有的硬件下是无法利用很多加速器的，所以实际使用起来并没有想象中那么快。所以作者引入了所谓的 **Fused-MBConv**。简单对比看可以发现，其实就是将 MBConv 模块的 $1 \times 1$ 和 DW conv 融合为了一个 $3 \times 3$ 卷积。实验发现在网络前期使用 Fused MBConv 会好一些，而不是无脑全部替换。最终作者使用 NAS 技术进行搜索，将前三个 MBConv 进行替换是最好的。

![img5](efficientnetv2-5.png)



`同等的放大每个 stage 是次优的`：在 EfficientNetV1 中每个 stage 的深度和宽度都是同等放大的。也就是直接简单粗暴的乘上宽度和深度缩放因子就行了。但是不同 stage 对于网络的训练速度，参数量等贡献并不相同，不能把他们同等看待。所以作者采用了非均匀的缩放策略进行缩放模型。但是作者并没有讲策略是什么，而是直接给出了参数。

![img6](efficientnetv2-6.png)



讲完 V1 之后，我们来看看 EfficientNetV2 做出的一系列贡献：

* 首先是引入了新的网络 EfficientNetV2，该网络在训练速度以及参数量上都优于先前的一些网络
* 提出了改进的渐近学习方法，该方法会根据训练图像的尺寸动态调整正则方法（Dropout、Rand Augment、Mixup），可以提升训练速度、准确率
* 通过实验与先前的一些网络进行对比，训练速度提升了 11 倍（EfficientNetV2-M 与 EfficientNet-B7 进行比较），参数数量减少为 1/6.8

![img7](efficientnetv2-7.png)



### 3. EfficientNetV2 网络框架

下表给出了 EfficientNetV2-S 的配置，一看和 EfficientNetV1 相近。但是有几个不同点：

* 除了使用 MBConv 之外还使用了 Fused-MBConv 模块
* 使用较小的 expansion ratio (之前是6)
* 偏向使用更小的 kernel\_size ($3 \times 3$， V1 中有 $5 \times 5$)
* 移除了 EfficientNetV1 中最后一个步距为 1 的 stage (V1 中的 stage8，注意是在 S 中删掉了)

在 stage0 中卷积后跟有 BN 和 SiLu 激活函数 ($silu(x)=x∗ sigmoid(x)$)，MBConv4 指的是主分支上第一个卷积层的扩展因子为 4。SE0.25 指 SE 模块第一个全连接层的节点个数是输入 MBConv 的特征矩阵通道数的 1/4。Layers 是重复的次数，stride 仅针对第一个 block。值得注意的是，源码中的配置和论文有所出入，在源码中 stage6 输出的 Channel 是 256 而不是表格中的 272，stage7 的输出 channel 是 1280 并非 1792。

![img8](efficientnetv2-8.png)



接下来我们更详细地看看这个 Fuse-MBConv 模块。论文中的图有 SE 模块，在源码实际搭建中其实是没有 SE 模块的。当 expansion = 1 的时候，在主分支上只有一个 $3 \times 3$ 卷积，跟上 BN 和 SILU 激活函数，以及一个 Dropout。而当 expansion > 1 的时候，在主分支上先有一个升维的 $3 \times 3$ 卷积，跟上 BN 和 SILU 激活函数，然后通过 $1 \times 1$ 卷积，然后经过 BN 和 Dropout。

Shortcut 连接只有在 stride=1 且输入特征矩阵 channel和主分支输出特征矩阵 channel 一样时才会有。Dropout 只有当存在 Shortcut 连接时才会有。

![img9](efficientnetv2-9.png)



BN 和 Dropout 一起用不是会有问题吗？所以这里的 Dropout 实际上不是以一定概率失活神经元 (nn.Drop2D)，而是 [Deep Networks with Stochastic Depth](https://arxiv.org/pdf/1603.09382.pdf)。以一定的概率**将主分支的输出完全丢弃**，即直接用上一层的输出。所以网络相当于是一个随机的深度了，因为没有这个 block 相当于我们的深度减少了“一层”嘛。在 EfficientNetV2 中失活概率是从 0-0.2 递增的。这样做的好处能提升训练速度，并小幅提升准确率。注意：这里的 dropout 层仅仅指 Fused-MBConv 和 MBConv 中的 Dropout，不包括最后全连接前的 Dropout。 

![img10](efficientnetv2-10.png)



紧接着我们看看源码中的配置如下：比如 `r2_k3_s1_e1_i24_o24_c1` 代表，Operator 重复堆叠 2 次，kernel_size 等于 3，stride 等于 1，expansion 等于 1，input_channels 等于 24，output_channels 等于 24，conv_type 为 Fused-MBConv。

![img11](efficientnetv2-11.png)

源码中关于解析配置的方法如下：

```python
  def _decode_block_string(self, block_string):
    """Gets a block through a string notation of arguments."""
    assert isinstance(block_string, str)
    ops = block_string.split('_')
    options = {}
    for op in ops:
      splits = re.split(r'(\d.*)', op)
      if len(splits) >= 2:
        key, value = splits[:2]
        options[key] = value

    return hparams.Config(
        kernel_size=int(options['k']),
        num_repeat=int(options['r']),
        input_filters=int(options['i']),
        output_filters=int(options['o']),
        expand_ratio=int(options['e']),
        se_ratio=float(options['se']) if 'se' in options else None,
        strides=int(options['s']),
        conv_type=int(options['c']) if 'c' in options else 0,
    )
```



源码中给的一些训练参数如下所示，例如对于 EfficientNetV2-S，训练输入尺寸为 300（最大训练尺寸为 300，实际训练的时候每张图是会变的），验证尺寸为固定的 384。这里 Dropout 对应的是最后全连接层之前的那个。后面的 randaug，mixup 以及 aug 则是针对渐进式学习使用到的超参数。

![img12](efficientnetv2-12.png)



### 4. 渐进式学习策略

作者做了一个实验，针对不同的训练输入尺寸，RandAug 在不同等级上取得了网络的最佳。那么作者就在想的，我们在使用不同训练输入尺寸训练时，需要使用不同的正则化方法呢？

![img13](efficientnetv2-13.png)



为此，作者提到，在训练早期的时候使用较小的训练尺寸以及较弱的正则化方法 weak regularization。这样网络能够很快的学习到一些简单的表达能力。而随着逐渐提升训练图像尺寸，同时增强正则化方法（adding stronger regularization），这里所说的 regularization 包括 `Dropout`，`RandAugment` 以及 `Mixup`。

![img14](efficientnetv2-14.png)



下表中给出了正则化强度如何随着图像尺寸变化，其实就是个线性插值（线性变化）。

![img15](efficientnetv2-15.png)



下表中给出了针对每一个模型使用的不同的图像训练尺寸以及正则化强度的变化范围。min 指 epoch 1，而 max 指 epoch 5。

![img16](efficientnetv2-16.png)



最后作者为了证明渐进式学习策略的有效性，作者在 ResNet 以及 EfficientNetV1 上也进行了实验，括号中是限定的最大尺寸。可见达到相同正确率的时候，训练时间大幅缩小，所以其是具有普适性的。

![img17](efficientnetv2-17.png)



**但是这些训练时间都是常人难以企及的啊！两张 3090 都吃不消...**

**学了这么多，感觉 block 本身的创意创新越来越少，学习方法和超参感觉对于最终的性能指标影响更大，很难自己训出来作者报告的指标，更难在论文中打败他们了......**



### 5. 代码

EfficientNetV2 实现代码如下所示，[代码出处](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/Test11_efficientnetV2/model.py)：

```python
from collections import OrderedDict
from functools import partial
from typing import Callable, Optional

import torch.nn as nn
import torch
from torch import Tensor


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvBNAct(nn.Module):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        super(ConvBNAct, self).__init__()

        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        self.conv = nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)

        self.bn = norm_layer(out_planes)
        self.act = activation_layer()

    def forward(self, x):
        result = self.conv(x)
        result = self.bn(result)
        result = self.act(result)

        return result


class SqueezeExcite(nn.Module):
    def __init__(self,
                 input_c: int,   # block input channel
                 expand_c: int,  # block expand channel
                 se_ratio: float = 0.25):
        super(SqueezeExcite, self).__init__()
        squeeze_c = int(input_c * se_ratio)
        self.conv_reduce = nn.Conv2d(expand_c, squeeze_c, 1)
        self.act1 = nn.SiLU()  # alias Swish
        self.conv_expand = nn.Conv2d(squeeze_c, expand_c, 1)
        self.act2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = x.mean((2, 3), keepdim=True)
        scale = self.conv_reduce(scale)
        scale = self.act1(scale)
        scale = self.conv_expand(scale)
        scale = self.act2(scale)
        return scale * x


class MBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(MBConv, self).__init__()

        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.has_shortcut = (stride == 1 and input_c == out_c)

        activation_layer = nn.SiLU  # alias Swish
        expanded_c = input_c * expand_ratio

        # 在EfficientNetV2中，MBConv中不存在expansion=1的情况所以conv_pw肯定存在
        assert expand_ratio != 1
        # Point-wise expansion
        self.expand_conv = ConvBNAct(input_c,
                                     expanded_c,
                                     kernel_size=1,
                                     norm_layer=norm_layer,
                                     activation_layer=activation_layer)

        # Depth-wise convolution
        self.dwconv = ConvBNAct(expanded_c,
                                expanded_c,
                                kernel_size=kernel_size,
                                stride=stride,
                                groups=expanded_c,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer)

        self.se = SqueezeExcite(input_c, expanded_c, se_ratio) if se_ratio > 0 else nn.Identity()

        # Point-wise linear projection
        self.project_conv = ConvBNAct(expanded_c,
                                      out_planes=out_c,
                                      kernel_size=1,
                                      norm_layer=norm_layer,
                                      activation_layer=nn.Identity)  # 注意这里没有激活函数，所有传入Identity

        self.out_channels = out_c

        # 只有在使用shortcut连接时才使用dropout层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        result = self.expand_conv(x)
        result = self.dwconv(result)
        result = self.se(result)
        result = self.project_conv(result)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)
            result += x

        return result


class FusedMBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 input_c: int,
                 out_c: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(FusedMBConv, self).__init__()

        assert stride in [1, 2]
        assert se_ratio == 0

        self.has_shortcut = stride == 1 and input_c == out_c
        self.drop_rate = drop_rate

        self.has_expansion = expand_ratio != 1

        activation_layer = nn.SiLU  # alias Swish
        expanded_c = input_c * expand_ratio

        # 只有当expand ratio不等于1时才有expand conv
        if self.has_expansion:
            # Expansion convolution
            self.expand_conv = ConvBNAct(input_c,
                                         expanded_c,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         norm_layer=norm_layer,
                                         activation_layer=activation_layer)

            self.project_conv = ConvBNAct(expanded_c,
                                          out_c,
                                          kernel_size=1,
                                          norm_layer=norm_layer,
                                          activation_layer=nn.Identity)  # 注意没有激活函数
        else:
            # 当只有project_conv时的情况
            self.project_conv = ConvBNAct(input_c,
                                          out_c,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          norm_layer=norm_layer,
                                          activation_layer=activation_layer)  # 注意有激活函数

        self.out_channels = out_c

        # 只有在使用shortcut连接时才使用dropout层
        self.drop_rate = drop_rate
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        if self.has_expansion:
            result = self.expand_conv(x)
            result = self.project_conv(result)
        else:
            result = self.project_conv(x)

        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)

            result += x

        return result


class EfficientNetV2(nn.Module):
    def __init__(self,
                 model_cnf: list,
                 num_classes: int = 1000,
                 num_features: int = 1280,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2):
        super(EfficientNetV2, self).__init__()

        for cnf in model_cnf:
            assert len(cnf) == 8

        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        stem_filter_num = model_cnf[0][4]

        self.stem = ConvBNAct(3,
                              stem_filter_num,
                              kernel_size=3,
                              stride=2,
                              norm_layer=norm_layer)  # 激活函数默认是SiLU

        total_blocks = sum([i[0] for i in model_cnf])
        block_id = 0
        blocks = []
        for cnf in model_cnf:
            repeats = cnf[0]
            op = FusedMBConv if cnf[-2] == 0 else MBConv
            for i in range(repeats):
                blocks.append(op(kernel_size=cnf[1],
                                 input_c=cnf[4] if i == 0 else cnf[5],
                                 out_c=cnf[5],
                                 expand_ratio=cnf[3],
                                 stride=cnf[2] if i == 0 else 1,
                                 se_ratio=cnf[-1],
                                 drop_rate=drop_connect_rate * block_id / total_blocks,
                                 norm_layer=norm_layer))
                block_id += 1
        self.blocks = nn.Sequential(*blocks)

        head_input_c = model_cnf[-1][-3]
        head = OrderedDict()

        head.update({"project_conv": ConvBNAct(head_input_c,
                                               num_features,
                                               kernel_size=1,
                                               norm_layer=norm_layer)})  # 激活函数默认是SiLU

        head.update({"avgpool": nn.AdaptiveAvgPool2d(1)})
        head.update({"flatten": nn.Flatten()})

        if dropout_rate > 0:
            head.update({"dropout": nn.Dropout(p=dropout_rate, inplace=True)})
        head.update({"classifier": nn.Linear(num_features, num_classes)})

        self.head = nn.Sequential(head)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)

        return x


def efficientnetv2_s(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 300, eval_size: 384

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[2, 3, 1, 1, 24, 24, 0, 0],
                    [4, 3, 2, 4, 24, 48, 0, 0],
                    [4, 3, 2, 4, 48, 64, 0, 0],
                    [6, 3, 2, 4, 64, 128, 1, 0.25],
                    [9, 3, 1, 6, 128, 160, 1, 0.25],
                    [15, 3, 2, 6, 160, 256, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.2)
    return model


def efficientnetv2_m(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[3, 3, 1, 1, 24, 24, 0, 0],
                    [5, 3, 2, 4, 24, 48, 0, 0],
                    [5, 3, 2, 4, 48, 80, 0, 0],
                    [7, 3, 2, 4, 80, 160, 1, 0.25],
                    [14, 3, 1, 6, 160, 176, 1, 0.25],
                    [18, 3, 2, 6, 176, 304, 1, 0.25],
                    [5, 3, 1, 6, 304, 512, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.3)
    return model


def efficientnetv2_l(num_classes: int = 1000):
    """
    EfficientNetV2
    https://arxiv.org/abs/2104.00298
    """
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[4, 3, 1, 1, 32, 32, 0, 0],
                    [7, 3, 2, 4, 32, 64, 0, 0],
                    [7, 3, 2, 4, 64, 96, 0, 0],
                    [10, 3, 2, 4, 96, 192, 1, 0.25],
                    [19, 3, 1, 6, 192, 224, 1, 0.25],
                    [25, 3, 2, 6, 224, 384, 1, 0.25],
                    [7, 3, 1, 6, 384, 640, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.4)
    return model

```

