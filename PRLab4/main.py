from net.SentimentAnalysis import SentimentAnalysis
from net.SinusoidalPrediction import SinusoidalPrediction


def main():
    sp = SinusoidalPrediction()
    sp.train()
    sa = SentimentAnalysis()
    sa.train()


if __name__ == '__main__':
    main()
