# Implementation of Transformer using PyTorch (detailed explanations )

This is a pytorch implementation of the Transformer model. If you'd like to learn more about the model, or any of the code better, please see my [https://say-hello2y.github.io/2022-08-18/transformer](https://say-hello2y.github.io/2022-08-18/transformer).

## Usages
### Step 1: install packages
```
pip install -r requirements.txt
```
### Step 2: train the model
```
python train.py
# use wandb (opt)
python train.py --track
'''
edit cond.py for better training
important parameters
epoch: training epoch
batch: batch size
'''

```
### Step 3 : test the model
```
python test.py
```

## Experiment

![](/saved/train_loss.png 'train loss during 100 epoch ')

![](/saved/valid_loss.png 'valid loss after each epoch during 100 epoch')