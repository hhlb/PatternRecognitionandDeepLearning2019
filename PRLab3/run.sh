#!/usr/bin/env bash
python main.py -d cuda:0 -n resnet -t -l 0.16 -o sgd
python main.py -d cuda:0 -n vgg -t -l 0.16 -o sgd

python main.py -d cuda:0 -n resnet -t -l 0.12 -o sgd
python main.py -d cuda:0 -n vgg -t -l 0.12 -o sgd

python main.py -d cuda:0 -n resnet -t -l 0.08 -o sgd
python main.py -d cuda:0 -n vgg -t -l 0.08 -o sgd

python main.py -d cuda:0 -n resnet -t -l 0.08 -o adam
python main.py -d cuda:0 -n vgg -t -l 0.08 -o adam

python main.py -d cuda:0 -n resnet -t -l 0.05 -o adam
python main.py -d cuda:0 -n vgg -t -l 0.001 -o adam