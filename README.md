# dl_concepts

# Perceptron

## A perceptron has 2 calculations to perform
* Dot product (z) of Features (X) with weights (w) (adding bias)
* Applying Activation function (a) to the Dot Product

### Model to learn AND gate logic
```python
python src/perceptron/perceptron.py
```

## There are different choices for Activation functions
* Sigmoid
* Hyperbolic tangent
* Relu (Relu and its variants Leaky Relu, Exponential LU, Parametric Relu)

### It is the Activation function, more precisely, different parts of activation function that gets activated with the "z" and results in fitting the  model to data
* Boundary line for simple linear problems
* Curves for non linear problems
* **Squiggles** for complex non linear problems

# Multi Layer Perceptron
* Perceptron is limited to solving basic problem, if we were to build a model that could imitate XOR gate, we need more than one perceptron
* So, in order to solve real world problems, we need different permutations and combinations of **Perceptron**, which results in MLP.
* MLP is the building block of complex Artificial Neural Network

<img src="https://i.stack.imgur.com/n2Hde.png" width=40% ali>

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

# Question: For ANNs, how do we find the weights and biases so that we get minimum value for Loss/Cost function?

## Solution: Backpropogation with chain rule

Weights and Bias update formula derived with the help of Chain rule

<img src='https://drive.google.com/uc?id=1jmL4SjzUwuv8xfiTo1sAWce5w8cfoGPT'>

### Gradient Descent
* The derivative term in above formula is the Gradient Descent
* As Derivative/Gradient (change in value of y with **small** change in value of x) gives direction  of **ASCENT**, hence in order to minimise loss we move in opposite direction therefore we reduce the original weight (negative sign in above formula)
* Different Optimizers can be tried to find this **Gradient** efficiently

### ANN Model on MNIST dataset
```python
python src/ann_mnist_basic_model/ann_mnist_basic_model.py
```

# Problems with ANN
* Vanishing Gradient: When using sigmoid activation function at Hidden layers
* Exploding Gradient: When weights are initialised with **High values**

## Activation Functions and Weight Initalisation Recommendation

<img src='https://drive.google.com/uc?id=17l9srUriABuaZ4JEKJGz7r2drMHFQ0jb'>

# Batch Normalisation
* To improve the training, we seek to reduce the **Internal Covariate Shift**. 
* By fixing the **distribution of the hidden layer inputs** as the training
progresses, we expect to improve the training speed. 
* When values are Normalised, GD converges faster and hence results in faster training

### ANN Model with Batch normalisation on MNIST dataset
```python
python src/batch_normalisation/bn_mnist.py
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
   * Batch size
   * etc..

## MlFlow
MlFLow helps by tracking different experiments that we can do with the various training parameters/ model metrics.
```python
python src/mlflow_ann_mnist/mlflow_ann_mnist.py
```
<img src='https://drive.google.com/uc?id=1RgU7w6c6Mnw9te7vooleeKY4ezUoqAA_'>

