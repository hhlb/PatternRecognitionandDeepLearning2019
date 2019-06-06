class ReadData(object):
    def __init__(self):
        self.keys = []
        self.values = []
        self.keys, self.values = self.readword()

    def readword(self):
        with open('./dataset/word.txt', 'r', encoding='UTF-8') as f:
            datas = f.read().split("\n")
            for i in range(len(datas)):
                d = datas[i].split(" ")
                self.keys.append(d[0])
                value = []
                for j in range(1, len(d)):
                    value.append(float(d[j]))
                self.values.append(value)
        return self.keys, self.values

    def findword(self, data):
        num = []
        for i in range(len(data)):
            if (data[i] in self.keys):
                j = self.keys.index(data[i])
                num.append(self.values[j])
            else:
                num.append(self.values[len(self.keys) - 2])
        return num

    def readneg(self):
        f = open('./dataset/rt-polarity-neg-unicode.txt', 'r', encoding='UTF-8')
        data = f.read()
        datas = data.split("\n")
        nvalues = []
        nkeys = []
        for i in range(len(datas)):
            d = datas[i].split(" ")
            value = []
            for j in range(len(d)):
                if d[j].find('\'s') > 0:
                    k = d[j].split("\'s")
                    value.append(k[0])
                    value.append("\'s")
                elif d[j] is not "":
                    value.append(d[j])
            num = self.findword(value)
            nvalues.append(num)
            nkeys.append([1, 0])
        f.close()
        return nkeys, nvalues

    def readpos(self):
        f = open('./dataset/rt-polarity-pos-unicode.txt', 'r', encoding='UTF-8')
        data = f.read()
        datas = data.split("\n")
        nvalues = []
        nkeys = []
        for i in range(len(datas)):
            d = datas[i].split(" ")
            value = []
            for j in range(len(d)):
                if d[j].find('\'s') > 0:
                    k = d[j].split("\'s")
                    value.append(k[0])
                    value.append("\'s")
                elif d[j] is not "":
                    value.append(d[j])
            num = self.findword(value)
            nvalues.append(num)
            nkeys.append([0, 1])
        f.close()
        return nkeys, nvalues
