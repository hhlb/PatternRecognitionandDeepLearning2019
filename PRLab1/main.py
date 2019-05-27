from mlp import run


def main():
  # 建立运行序列 初始化参数和函数
  se = run.Sequence()
  # 运行训练序列
  se.train()
  # 运行测试
  se.test()


if __name__ == '__main__':
  main()
