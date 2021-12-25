---
title: 论文阅读《Heterogeneous Face Attribute Estimation:A Deep Multi-Task Learning Approach》
date: 2018/10/23 17:10:00
tags: face attributes
mathjax: true
---

### 简介
本文介绍了一个用于人脸分类的文章，该文章发表在2018年的《TPAMI》期刊上。
文章主要做了如下几个工作：
- 避开人脸姿态、对齐的工作，着重对多种属性的联合预估进行分析判断。
- 充分使用人脸属性的相关性与异质性进行训练。
- 端到端的学习训练模式。
- 使用大量的场景进行训练。
{% asset_img attribute_heterogeneity.png 人脸属性的相关性与异质性 %}

### 主要思路
本文经过分析，将人脸的属性按照数据类型和语义信息这两种方式进行了划分，分别是全局与局部属性，连续与离散属性。经过排列组合，可以得到四类属性分组。
对于每一类属性，网络共享前置的基础网络特征，在基础网络后针对不同的分组，分别设置了相对独立的不同的神经网络结构。

| | |
| :------: | :------: |
| ordinal + local | nominal + local |
| ordinal + global | nominal + global |

#### 损失函数
##### 多任务学习
传统的多任务学习使用如下方式：

$$
{argmin}_{W_{j=1}^M} \sum_{j=1}^{M} \sum_{i=1}^{N} L(y^i_j, F(X_i,W^j)) + \gamma\Phi(W^j) )
$$

作者在此基础上引入了对共享权值和独立权值的学习率参数，以区别对待。

$$
{argmin}_{W_c, W_{j=1}^M} \sum_{j=1}^{M} \sum_{i=1}^{N} 
L(y^i_j, F(X_i,W^j \circ W_c)) + \gamma_1\Phi(W_c) ) + \gamma_2\Phi(W^j) )
$$

针对分组多任务学习，损失函数相应地调整为：

$${argmin}_{W_c, W_{j=1}^M} \sum_{g=1}^G \sum_{j=1}^{M^g} \sum_{i=1}^{N} 
\lambda^g L^g(y^i_j, F(X_i,W^g \circ W_c)) + \gamma_1\Phi(W_c) ) + \gamma_2\Phi(W^g) )$$

##### 离散特征 VS 连续特征
- 对于连续属性特征，模型使用L2距离进行回归训练；
$$
L^{gO}=\sum_{j=1}^{M^o}\sum_{i=1}^{N}{\left|\left|{y_i^j-\hat{y}^j_i}\right|\right|}^2_2
$$
- 对于离散属性特征，模型采用交叉熵进行模型训练。
$$
L^{gN} = -\sum_{j=1}^{M^N} \sum_{i=1}^N \sum_{k=1}^{C^j} 1(yi^j_, \hat{y}^{j,k}_i) log p(\hat{y}^{j,k}_i)
$$

#### 网络模型
本文的网络结构在AlexNet的基础上做了一些修改，包含一个5层的卷积和2层全连接，每一层卷积都跟着一个BN层和一个最大池化层。经过这7层的网络学习后得到共享的属性特征。接着对于不同组别的属性分类任务，模型分别使用一个独立的网络结构进行特征学习。
{% asset_img framework.png 论文使用的网络结构 %}


### 实验部分
文章在MORPH II、CelebA、LFWA、ChaLearn LAP和FotW等数据集上进行了验证。

#### 数据预处理
实验中使用SeetaFace Engine对数据集中的图像进行人脸检测和特征点定位，随后基于5个特征点将人脸归一化到$256 \ast 256 \ast 3$的尺寸。

对于属性分组，本文将所有属性按照数据类型（离散、连续）与语义信息（全局、局部）分成了四类。

#### 连续与离散属性预测
{% asset_img attribute_accuracy_morph_lfw.jpg MORPH和LFW数据集上的实验效果 %}

#### 二元属性预测
与主流算法在CelebA和LFWA数据集上的比较都达到了比较理想的效果。
{% asset_img attribute_accuracy_celeba_lfwa.jpg CelebA和LFWA数据集上的实验效果 %}

#### 单任务与多任务对比
实验利用CelebA中常见的8中属性进行多任务训练与单任务训练的对比。实验结果表明在大部分场景下，使用多任务能够显著提升模型性能。
{% asset_img accuracy_multi_task.jpg 单任务与多任务实验的效果对比 %}

#### 跨数据集测试
论文进行了跨数据集的实验以验证模型的泛化能力。
{% asset_img cross_dataset_accuracy.jpg 跨数据集测试的实验效果 %}

#### 运行效率分析
算法的运行效率如图所示，相比主流方法有较为明显的优势。
{% asset_img computational_cost.jpg 算法运行效率 %}

### 文献引用
- Han H, Jain A K, Wang F, et al. Heterogeneous face attribute estimation: A deep multi-task learning approach[J]. IEEE transactions on pattern analysis and machine intelligence, 2017, 40(11): 2597-2609.
