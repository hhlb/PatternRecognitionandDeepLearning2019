from scipy.io import loadmat


class Data(object):
    def __init__(self):
        super(Data, self).__init__()
        m = loadmat('./mat/points.mat')
        self.mat_data = m['xx']
