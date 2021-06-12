# DL concepts

* [Perceptron](#perceptron)
* [Multi Layer Perceptron](#multi-layer-perceptron)
* [Activation functions](#activation-functions)
* [Cost function](#cost-function)
* [Optimizers](#optimizers)
* [Backpropogation with chain rule](#backpropogation-with-chain-rule)
* [Problems with ANN](#problems-with-aNN)
* [Batch Normalisation](#batch-normalisation)
* [MNIST Dataset](#mNIST-dataset)
* [Tuning Model](#tuning-model)
* [MlFlow](#mlFlow)
* [Transfer Learning](#transfer-learning)

# Perceptron

## A perceptron has 2 calculations to perform
* Dot product (z) of Features (X) with weights (w) (adding bias)
* Applying Activation function (a) to the Dot Product

### Model to learn AND gate logic
```python
python src/perceptron/perceptron.py
```

# Multi Layer Perceptron
* Perceptron is limited to solving basic problem, if we were to build a model that could imitate XOR gate, we need more than one perceptron
* So, in order to solve real world problems, we need different permutations and combinations of **Perceptron**, which results in MLP.
* MLP can be considered as a subset of Artificial Neural Networks or Deep Neural Networks

<img src="https://i.stack.imgur.com/n2Hde.png" width=40% ali>

# Activation functions
There are different choices for Activation functions
* Sigmoid
* Hyperbolic tangent
* Relu (Relu and its variants Leaky Relu, Exponential LU, Parametric Relu)
* Softmax
* Maxout (ReLU and leaky ReLU are special cases of Maxout)
* Swish
* Softplus

## It is the Activation function, more precisely, different parts of activation function that gets activated with the "z" and results in fitting the  model to data
* Boundary line for simple linear problems
* Curves for non linear problems
* **Squiggles** for complex non linear problems

## Dying ReLu
* Dying ReLu can occure if your dataset contains negative values.
* Negative values results in zero acivation hence no weight update.

## Gradient Saturation
* Gradient Saturation is caused when you use sigmoid or tanh like functions
* sigmoid or tanh function have almost zero gradient on their **ends** (refer graph to understand this). 
* Gradient staturation means **gradient tends to zero** which results into zero or neglegible weight updates.  

# Cost function
* Now we have Output from MLP (**y hat**), here onwards ANN and we also know the Actual Output (**y**).
* Using y hat and y, we calculate error/loss made by the model/network using **Cost functions**
* Options for Cost functions depends on problem to solve
  * Regression
    * MAE (L1) /MSE (L2) / RMSE
  * Classification
    * CrossEntropy or Log Loss /Hinge Loss

# Optimizers
To minimise loss we perform **Gradient Descent**. The entity that performs gradient descent is termed as **Optimizer**
* Optimizers choices that works purely based on **previous weights**
  * SGD Momentum
  * ADAM (Adaptive Moment Estimation)
* Optimizers choices that also considers **previous Learning Rates**
  * AdaGrad
  * AdaGradDelta
  * RMSPROP 

# Backpropogation with chain rule
## Question: For ANNs, how do we find the weights and biases so that we get minimum value for Loss/Cost function?
## Solution: Backpropogation with chain rule

Weights and Bias update formula derived with the help of Chain rule

<img src='https://drive.google.com/uc?id=1jmL4SjzUwuv8xfiTo1sAWce5w8cfoGPT'>

### Gradient Descent
* The derivative term in above formula is the Gradient Descent
* As Derivative/Gradient (change in value of y with **small** change in value of x) gives direction  of **ASCENT**, hence in order to minimise loss we move in opposite direction **(Descent)** therefore we reduce the original weight (negative sign in above formula)
* Different Optimizers can be tried to find this **Gradient** efficiently

# Problems with ANN
* Vanishing Gradient: When using sigmoid activation function at Hidden layers
* Exploding Gradient: When weights are initialised with **High values**

## Activation Functions and Weight Initalisation Recommendation

<img src='https://drive.google.com/uc?id=17l9srUriABuaZ4JEKJGz7r2drMHFQ0jb'>

# Batch Normalisation
* To improve the training, we seek to reduce the **Internal Covariate Shift**. 
* By fixing the **distribution of the hidden layer inputs** as the training progresses, we expect to improve the training speed. 
* When values are Normalised, GD converges faster and hence results in faster training
* Batch Normalisation introduces 4 parameters out of which only 2 are learnable parameters *(2 Extra trainable parameters)
* When uisng Batch Normalisation **before Activation**, then there is **no need to use bias** as becasue BN layer itself contains Beta term which is equivalent to Bias.
* When uisng Batch Normalisation **after Activation**, then you need to use bias as activation function generally prefer input containing weights and biases.
* Batch Normalisation is useful when we have deep neural networks (CNN)
* Recommendation: When our network has more than **16** layers use Batch Normalisation

### ANN Model with Batch normalisation on MNIST dataset
```python
python src/batch_normalisation/bn_mnist.py
```

# MNIST Dataset
* Each image is a 28*28 matrix 
* To feed them in Neural network we have to make them in a single row or array, so we perform **Flatten** operation
* 60,000 images are fed to ANN as 60,000 arrays. 
* With **CNN** we directly feed the image without flattening operation.

## ANN Model on MNIST dataset
```python
python src/ann_mnist_basic_model/ann_mnist_basic_model.py
```

# Tuning Model
Tuning a particular model can be splitted as below
* Architecture of Model: 
   * Number of Hidden Layers in the model, 
   * Number of Neurons at a Hidden layers
   * Activation function at Hidden layers
   * Model with Batch Normalisation
* Model Metrics: 
   * Optimizer
   * Cost/Loss Function
* Training Parameters: 
   * Epochs
   * Batch size (As all the data connot be fed at once in the neural network while training, because of the RAM memory constraints we feed data into batches)
   * etc..

# MlFlow
MlFLow helps by tracking different experiments that we can do with the various training parameters and model metrics.
```python
python src/mlflow_ann_mnist/mlflow_ann_mnist.py
```
<img src='https://drive.google.com/uc?id=1l_0Fxx8jC-MrZsVWFPU7feZH00925BGm'>

Starting from bottom: 

* Run 1: batch_size: 50, epoch 1, Activation function: relu, Optimizer: SGD - Accuracy: 0.809
* Run 2: batch_size: 50, epoch **20**, Activation function: relu, Optimizer: SGD - Accuracy: **0.977**
* Run 3: batch_size: **100**, epoch 20, Activation function: relu, Optimizer: SGD - Accuracy: 0.959
* Run 4: batch_size: 100, epoch 20, Activation function: relu, Optimizer: **Adam** - Accuracy: **0.985**
* Run 5: batch_size: 100, epoch 20, Activation function: relu, Optimizer: Adam, **batch normalisation** - Accuracy: **0.994**
* Run 6: batch_size: 100, epoch 20, Activation function: relu, Optimizer: Adam, batch normalisation with **Bias as false** - Accuracy: 0.993
* Run 7: batch_size: 100, epoch 20, Activation function: **sigmoid**, Optimizer: SGD - Accuracy: 0.922
* Run 8: batch_size: 100, epoch 20, Activation function: sigmoid, Optimizer: **Adam** - Accuracy: **0.997**
* Run 9: **Weight initialization**: Change relu with **he_normal**
  batch_size: 100, epoch 20, Activation function: relu, Optimizer: **Adam** - Accuracy: **0.998 Best so far**
  
  ## MlFlow experiments with Optimizers
  Constant Parameters: batch_size: 100, epoch 20, Activation function: sigmoid, kernel_initializer: glorot_normal
  
  * Run 10: learning_rate: 0.001: 100, momentum: 0.0, nesterov: False - Accuracy: 0.561
  * Run 11: learning_rate: 0.001: 100, **momentum**: 0.9, nesterov: False - Accuracy: 0.872
  * Run 12: learning_rate: 0.001: 100, momentum: 0.9, **nesterov**: True - Accuracy: 0.871
  
  Constant Parameters: batch_size: 100, epoch 20, Activation function: **relu**, kernel_initializer: he_normal
  * Run 13: learning_rate: 0.001: 100, momentum: 0.0, nesterov: False - Accuracy: 0.898
  * Run 14: learning_rate: 0.001: 100, **momentum**: 0.9, nesterov: False - Accuracy: 0.961
  * Run 15: learning_rate: 0.001: 100, momentum: 0.9, **nesterov**: True - Accuracy: 0.963
 
* Run 10: Loss functions: Above observations are with **sparse_categorical_crossentropy**
* Run 11: Regularisation techniques 
  * L1:
  * L2:
  * Dropout techniques:

## Observation on Early Stopping and Check-pointing
```python
accuracy          loss          restored_epoch          stopped_epoch
0.9975454807	0.01011859719	          9	               14
0.9937454462	0.02368656360	         11	               16
0.9965272546	0.01463884022	         13	               18
0.9928908944	0.02340739779	          5	               10
0.9936727285	0.01931540295	          8	               13
0.9849818349	0.04650410637	          3	                8
```

### After certain epochs if Accuracy/Loss would not improve, the training will be stopped even before desired number of epochs (2)

# Transfer Learning
* Transfer learning enables us to use the already built robust models
* Steps for Transfer learning
  * Step 1: Load Previous model
  * Step 2: Check Model Details (available Trainable layers)
  * Step 3: Remove Last Layer
  * Step 4: Create new model
  ```python
    lower_pretrained_layers = model.layers[:-1]

    new_model = tf.keras.models.Sequential(lower_pretrained_layers)
    new_model.add(
        tf.keras.layers.Dense(2, activation="softmax", name="NewOutputLayer")
        # tf.keras.layers.Dense(2, activation="binary")
    )
  ```
* Train the new Model which now already has weights for older layers
* We will train it to get weights for new layer(s) added to the model to suit our use case (example: Is the numer Even?)
```python
python src/transfer_learning/is_even.py
```

* Latex: https://latex.codecogs.com/eqneditor/editor.php
* Netron: https://netron.app/
