from Nets.trian import GanNetsTT, WganNetsTT, WganGPNetsTT


def main():
    a = GanNetsTT()
    a.gan_train()
    b = WganNetsTT()
    b.wgan_train()
    c = WganGPNetsTT()
    c.wgangp_train()


if __name__ == '__main__':
    main()
