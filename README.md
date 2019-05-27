# PatternRecognitionandDeepLearning2019
哈尔滨工业大学模式识别与深度学习2019实验

本实验是2019年哈尔滨工业大学计算机科学与技术视听觉信号处理方向的专业课实验，所有的文件并没有进行详细的注释。

## 运行环境

本实验的环境建议通过`anaconda3`进行部署

```
python=3.7
pytorch=1.1
cuda=10.0
```

## 实验

### PRLab2 AlexNet

实验手动搭建AlexNet在CIFA10数据集上进行训练并通过tensorboardX来可视化展现
```
.
├── AlexNet #存放AlexNet的相关文件
│   ├── Dataset.py  #下载并处理数据集
│   ├── __init__.py
│   ├── Module.py #进行训练和测试
│   └── Net.py #搭建网络
├── dataset #存放数据集
│   ├── cifar-10-batches-py
│   │   ├── batches.meta
│   │   ├── data_batch_1
│   │   ├── data_batch_2
│   │   ├── data_batch_3
│   │   ├── data_batch_4
│   │   ├── data_batch_5
│   │   ├── readme.html
│   │   └── test_batch
│   └── cifar-10-python.tar.gz
├── main.py #调度Module的训练和测试
├── result1.pkl #模型 不上传
└── runs #tensorboard的输出
    └── May22_18-48-42_hhlb-MS-7B38
        ├── events.out.tfevents.1558522122.hhlb-MS-7B38
        └── train
            ├── acc
            │   └── events.out.tfevents.1558522125.hhlb-MS-7B38
            └── loss
                └── events.out.tfevents.1558522125.hhlb-MS-7B38
```

### PRLab3 VGG11 ResNet