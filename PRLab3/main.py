import argparse

import net


def main(args):
    import settings
    net.Run(settings.net, settings.device, settings.lr, args.train, settings.optimizer)


if __name__ == '__main__':
    n = ('resnet', 'vgg')
    o = ('sgd', 'adam')
    parser = argparse.ArgumentParser(usage='通过指定网络、设备和参数进行基于CIFA-10的物体分类训练。',
                                     description='通过以下的参数来进行设置，但是请遵守给定的数据范围，保证合法数据的使用。')
    parser.add_argument('-d', '--device', help='指定设备进行训练，如果是GPU请设置为 cuda:x 来指定使用x号GPU。', default='cpu')
    parser.add_argument('-n', '--net', help='指定学习网络', default='resnet')
    parser.add_argument('-lr', '--learningrate', help='学习率', default=0.1)
    parser.add_argument('-t', '--train', help='重新训练网络', action='store_true')
    parser.add_argument('-o', '--optimizer', help='优化器选择', default='SGD')
    args = parser.parse_args()
    so = str(args.optimizer).lower()
    s = str(args.net).lower()
    if (s not in n) or (so not in o):
        print('Net must be', n)
        print('Optimizer must be', o)
        exit(0)
    with open('settings.py', 'w') as f:
        f.write('device=\'' + str(args.device).lower() + '\'\n')
        f.write('net=\'' + str(args.net).lower() + '\'\n')
        f.write('lr=' + str(args.learningrate) + '\n')
        f.write('optimizer=\'' + str(args.optimizer) + '\'\n')

    try:
        import settings

        print('device:', settings.device)
        print('net', settings.net)
        print('lr', settings.lr)
        print('retrain', args.train)
        print('optimizer', settings.optimizer)
    except:
        print('There is something wrong with settings.py.')
        exit(0)
    main(args)
