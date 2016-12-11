# What's this
Implementation of FractalNet by chainer  

# Dependencies

    git clone https://github.com/nutszebra/fractal_net.git
    cd fractal_net
    git submodule init
    git submodule update

# How to run
    python main.py -g 0

# Note
Learning is stagnated until 20 epoch, so be patient


# Details about my implementation
All hyperparameters and network architecture are the same as in [[1]][Paper] except for data-augmentation and learning rate schdedule.  

* Data augmentation  
Train: Pictures are randomly resized in the range of [32, 36], then 32x32 patches are extracted randomly and are normalized locally. Horizontal flipping is applied with 0.5 probability.  
Test: Pictures are resized to 32x32, then they are normalized locally. Single image test is used to calculate total accuracy.  

* Learning rate schedule
Initial leranig rate is 0.02 and it is divided by 10 at [150, 225, 300, 375] epoch.

# Cifar10 result
| network                   | total accuracy (%) |
|:--------------------------|-------------------:|
| FractalNet [[1]][Paper]   | 95.41              |
| my implementation         | soon               |

<img src="https://github.com/nutszebra/fractal_net/blob/master/loss.jpg" alt="loss" title="loss">
<img src="https://github.com/nutszebra/fractal_net/blob/master/accuracy.jpg" alt="total accuracy" title="total accuracy">

# References
FractalNet: Ultra-Deep Neural Networks without Residuals [[1]][Paper]

[paper]: https://arxiv.org/abs/1605.07648 "Paper"
