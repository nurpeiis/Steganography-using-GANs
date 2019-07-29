# Steganography for sequential data
The main aim of this project is to test double layer of encoding and LeakGAN and RNN models for sequential steganographic system. In particular we have concentrated on 
generating steganographic text, as a test case of sequential data. 

## How Double Layer of Encoding Works

## Requirements
* **PyTorch r1.1.0**
* Python 3.5+
* CUDA 8.0+ (For GPU)

## Training 

To start training LeakGAN model on EMNLP data run following commands:

`cd run`

`python3 run-leakgan.py 3`

To start the training LSTM model on EMNLP data run following commands:

`cd LSTM`

`python3 main.py`

## Generation 

To generate stego text using EMNLP_NEWS dataset & and double layer of encoding run following commmand:

`python3 generate.py 2`


## Reference
You can find more details on how LeakGAN model works in following paper:
```bash
@article{guo2017long,
  title={Long Text Generation via Adversarial Training with Leaked Information},
  author={Guo, Jiaxian and Lu, Sidi and Cai, Han and Zhang, Weinan and Yu, Yong and Wang, Jun},
  journal={arXiv preprint arXiv:1709.08624},
  year={2017}
}
```

One of the research works that considers similar theme can be found using this [link](https://github.com/tbfang/steganography-lstm)

## Acknoledgements:
1. [LeakGAN Model](https://github.com/williamSYSU/TextGAN-PyTorch) was mainly using this resource
2. [RNN](https://github.com/pytorch/examples/tree/master/word_language_model)

Copyright (c) 2019 Nurpeiis Baimukan
