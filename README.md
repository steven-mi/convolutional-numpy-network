# convolutional-neural-network-with-numpy
A convolutional neural network implemented with python and only numpy. This is also my assigment for the course "B55.2 WT Ausgewählte Kapitel sozialer Webtechnologien" at "HTW-Berlin"

## Getting Started
These instructions will get you a copy of the project up and running a simple convolutional neural network on your local machine.

### Prerequisites
Make sure numpy is installed. Below should work
```
python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
```
Also your data shape would have the shape
```
(m, c, w, h) e.g (5000,1,50,50) 5000 50x50 images with 1 channel
```
with m beeing the number of images, c beeing the color channel and w, h beeing width and height. Note that this network will only run with squared images so width has to be equal to height

### Building a model
You have to define an custom network architecture by stacking layers and activation functions. Currently following layers and loss functons are implemented

#### Activation functions
* ReLU
* Leaky ReLU
* Parametric ReLU
* Exponential ReLU
* Tanh
* Sigmoid

#### Layers
* Flatten
* Fully Connected Layers
* Convolution layer
* Pooling layer
* Dropout layer

#### Loss functions
* Cross entropy with softmax loss

with these options in mind you can create a convolutional neural network e.g.
```python
def fcn():
    conv_01 = Conv(1, 32, (3, 3), stride=1, padding=0)
    relu_01 = ReLU()
    conv_02 = Conv(32, 32, (3, 3), stride=1, padding=0)
    relu_02 = ReLU()
    max_pool_01 = Pool()
    conv_03 = Conv(32, 64, (3, 3), stride=1, padding=0)
    relu_03 = ReLU()
    conv_04 = Conv(64, 64, (3, 3), stride=1, padding=0)
    relu_04 = ReLU()
    max_pool_02 = Pool()
    dropout_01 = Dropout(0.25)
    flat = Flatten()
    hidden_01 = FullyConnected(5184, 128)
    relu_1 = ReLU()
    dropout_02 = Dropout()
    output = FullyConnected(128, 7)
    return [conv_01, relu_01, conv_02, relu_02, max_pool_01, conv_03, relu_03, conv_04, relu_04, max_pool_02, dropout_01,
    flat, hidden_01, relu_1, dropout_02, output]

fcn = NeuralNetwork(fcn, score_func=LossCriteria.softmax)
```

#### Training the model
After defining a model you can start training it by selecting one of the optimizers and set up their hyperparametres.
```python
fcn = Optimizer.adam(fcn, X_train, y_train, LossCriteria.mean_squared, batch_size=64,
                    epoch=50, learning_rate=0.001, X_test=X_test, y_test=y_test, verbose=True)
```
#### Optimizers
* SGD
* SGD + momentum
* SGD + nesterov momentum
* AdaGrad
* AdaDelta
* RMSProp
* Adam
