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

### PRLab1 MLP

实验通过mlp进行手写数据识别
```
.
├── dataset #存放数据集
├── main.py
└── mlp
    ├── __init__.py
    ├── mlp.py  #搭建网络
    └── run.py  #测试运行
```


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

实验手动搭建resnet18和vgg11进行CIFA10的分类，其中数据集需要手动加载
```
.
├── main.py
├── net
│   ├── __init__.py
│   ├── net.py
│   ├── resnet.py
│   └── vgg.py
├── readdata
│   ├── __init__.py
│   ├── mydata.py
│   └── path.py
├── README.md
├── runs
│   ├── May28_11-22-56_hhlb-MS-7B38
│   │   └── events.out.tfevents.1559013776.hhlb-MS-7B38
│   ├── May28_11-24-43_hhlb-MS-7B38
│   │   └── events.out.tfevents.1559013883.hhlb-MS-7B38
│   ├── May28_11-26-18_hhlb-MS-7B38
│   │   └── events.out.tfevents.1559013978.hhlb-MS-7B38
│   ├── May28_11-28-08_hhlb-MS-7B38
│   │   └── events.out.tfevents.1559014088.hhlb-MS-7B38
│   ├── May28_11-29-43_hhlb-MS-7B38
│   │   └── events.out.tfevents.1559014183.hhlb-MS-7B38
│   ├── May28_11-31-36_hhlb-MS-7B38
│   │   └── events.out.tfevents.1559014296.hhlb-MS-7B38
│   ├── May28_11-35-07_hhlb-MS-7B38
│   │   └── events.out.tfevents.1559014507.hhlb-MS-7B38
│   ├── May28_11-37-05_hhlb-MS-7B38
│   │   └── events.out.tfevents.1559014625.hhlb-MS-7B38
│   ├── May28_11-42-21_hhlb-MS-7B38
│   │   └── events.out.tfevents.1559014941.hhlb-MS-7B38
│   └── May28_11-50-11_hhlb-MS-7B38
│       └── events.out.tfevents.1559015411.hhlb-MS-7B38
├── run.sh
└── settings.py
```

**Lab3使用说明**：`python main.py -h`

### PRLab4 RNN

编写RNN网络实现函数预测，使用影评数据进行文本情感分析。

```
.
├── dataset
│   ├── rt-polarity-neg-unicode.txt
│   ├── rt-polarity-pos-unicode.txt
│   ├── rt-polarity.neg
│   ├── rt-polarity.pos
│   └── word.txt
├── main.py
├── mydata
│   ├── __init__.py
│   └── dataset.py
├── net
│   ├── Net.py
│   ├── SentimentAnalysis.py
│   ├── SentimentAnalysisSettings.py
│   ├── SinusoidalPrediction.py
│   └── SinusoidalPredictionSettings.py
└── report
```

### PRLab5 GAN

编写GAN WGAN WGAN-GP实现分布拟合

```
.
├── Nets
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-37.pyc
│   │   ├── dg.cpython-37.pyc
│   │   ├── settings.cpython-37.pyc
│   │   └── trian.cpython-37.pyc
│   ├── dg.py
│   ├── settings.py
│   └── trian.py
├── main.py
├── mat
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-37.pyc
│   │   └── data.cpython-37.pyc
│   ├── data.py
│   └── points.mat
├── report
└── results
```