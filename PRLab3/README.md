# 使用说明

```shell
python main.py -h #帮助信息
usage: 通过指定网络、设备和参数进行基于CIFA-10的物体分类训练。

通过以下的参数来进行设置，但是请遵守给定的数据范围，保证合法数据的使用。

optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
                        指定设备进行训练，如果是GPU请设置为 cuda:x 来指定使用x号GPU。
  -n NET, --net NET     指定学习网络
  -lr LEARNINGRATE, --learningrate LEARNINGRATE
                        学习率
  -t, --train           重新训练网络
  -o OPTIMIZER, --optimizer OPTIMIZER
                        优化器选择
```