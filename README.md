# Deep Learning

* [Perceptron](#perceptron)
* [Multi Layer Perceptron](#multi-layer-perceptron)
* [Problems with ANN](#problems-with-ann)
* [Tuning Model](#tuning-model)
* [MlFlow Experiments](#mlflow-experiments)
  * [MNIST Dataset](#mnist-dataset) 
  * [Activation functions](#activation-functions)
  * [Weight Initalisation Recommendation](#weight-initalisation-recommendation)
  * [Batch Normalisation](#batch-normalisation)
  * [Cost function](#cost-function)
  * [Optimizers](#optimizers)
    * [Backpropogation with chain rule](#backpropogation-with-chain-rule)
    * [Types of optimisers](#types-of-optimisers)
  * [Regularisation Techniques](#regularisation-techniques)
  * [Observation on Early Stopping and Check-pointing](#observation-on-early-stopping-and-check-pointing)
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

# Problems with ANN
* Vanishing Gradient: When using sigmoid activation function at Hidden layers
* Exploding Gradient: When weights are initialised with **High values**

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

**Let's understand above by performing experiments with MlFlow**

# MlFlow Experiments
MlFLow helps by tracking different experiments that we can do with the various training parameters and model metrics.
```
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts/mlflow-artifacts --host 0.0.0.0 -p 1234
```
## MNIST Dataset
* Each image is a 28*28 matrix (number 0-9)
* To feed them in Neural network we have to make them in a single row or array, so we perform **Flatten** operation
* 60,000 images are fed to ANN as 60,000 arrays. 
* With **CNN** we directly feed the image without flattening operation.

## ANN Model on MNIST dataset
```python
python src/ann_mnist_basic_model/ann_mnist_basic_model.py
```
```python
python src/mlflow_ann_mnist/mlflow_ann_mnist.py
```
### MlFLow experiments with Epoch and Batch size
* Run 1: batch_size: 50, epoch 1, Activation function: relu, Optimizer: SGD - Accuracy: 0.809
* Run 2: batch_size: 50, epoch **20**, Activation function: relu, Optimizer: SGD - Accuracy: **0.977**
* Run 3: batch_size: **100**, epoch 20, Activation function: relu, Optimizer: SGD - Accuracy: 0.959

## Activation functions
There are different choices for Activation functions
* Sigmoid
* Hyperbolic tangent
* Relu (Relu and its variants Leaky Relu, Exponential LU, Parametric Relu)
* Softmax
* Maxout (ReLU and leaky ReLU are special cases of Maxout)
* Swish
* Softplus
### It is the Activation function, more precisely, different parts of activation function that gets activated with the "z" and results in fitting the  model to data
* Boundary line for simple linear problems
* Curves for non linear problems
* **Squiggles** for complex non linear problems

### Dying ReLu
* Dying ReLu can occure if your dataset contains negative values.
* Negative values results in zero acivation hence no weight update.

### Gradient Saturation
* Gradient Saturation is caused when you use sigmoid or tanh like functions
* sigmoid or tanh function have almost zero gradient on their **ends** (refer graph to understand this). 
* Gradient staturation means **gradient tends to zero** which results into zero or neglegible weight updates. 

### MlFLow experiments with Activation function
* Run 1: batch_size: 100, epoch 20, Activation function: **sigmoid**, Optimizer: SGD - Accuracy: 0.922

## Weight Initalisation Recommendation
| Initialisation  | Activation function              |
| --------------- | -------------------------------- |
| Glorot          | None, Tanh, Sigmoid and Softmax  |
| He              | Relu and its variants            |
| LeCum           | Selu                             |

### MlFLow experiments with Weight Initalisation
* Run 1: batch_size: 100, epoch 20, Activation function: **sigmoid**, Optimizer: SGD - Accuracy: 0.922
* Run 9: **Weight initialization**: Change relu with **he_normal**

## Batch Normalisation
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
### MlFLow experiments with Batch Normalisation
* Run 5: batch_size: 100, epoch 20, Activation function: relu, Optimizer: Adam, **batch normalisation** - Accuracy: **0.994**
* Run 6: batch_size: 100, epoch 20, Activation function: relu, Optimizer: Adam, batch normalisation with **Bias as false** - Accuracy: 0.993

## Cost function
* We now have the output from ANN (**y hat**) and we also know the Actual Output (**y**).
* Using y hat and y, we calculate error/loss made by the model/network using **Cost functions**
* Options for Cost functions depends on problem to solve
  * Regression
    * MAE (L1) /MSE (L2) / RMSE
  * Classification
    * CrossEntropy or Log Loss /Hinge Loss
 
## MlFlow experiments with Loss Functions 
Above observations are with **sparse_categorical_crossentropy**

## Optimizers
To minimise loss we perform **Gradient Descent**. The entity that performs gradient descent is termed as **Optimizer**
### Backpropogation with chain rule
#### Question: For ANNs, how do we find the weights and biases so that we get minimum value for Loss/Cost function?
#### Solution: Backpropogation with chain rule
* Weights and Bias update formula derived with the help of Chain rule
* Consider below Neural Network

<img src="https://github.com/sbhrwl/dl_experiments/blob/main/artifacts/images/GD.jpg" width="1000"/>

  * x1: Input to the NN
  * a0: Activation at Buffer layer (No activation)
  * w1: weight Initialised for Hidden layer 1
  * z1: The dot product of weight and inputs (adding bias) for Hidden layer 1
  * a1: Output of Hidden layer after **applying activation** to the dot product of weight and inputs (adding bias)
  * w2: weight Initialised for Hidden layer 2
  * z2: The dot product of weight and inputs (adding bias) for Hidden layer 2
  * **a2**: Output of Hidden layer after **applying activation** to the dot product of weight and inputs (adding bias)
  * a2 is **yhat** output of the Neural network

### Cost function for the NN as a function of w2
<img src="https://render.githubusercontent.com/render/math?math=Error(e) = (y - \hat{y})^{2} = (y - a_{2})^{2} \rightarrow f(a_{2})">

<img src="https://render.githubusercontent.com/render/math?math=a_{2} = \sigma (z_{2}) \rightarrow f(z_{2})">
<img src="https://render.githubusercontent.com/render/math?math=z_{2} = w_{2}a_{1} \rightarrow f(w_{2}, a_{1})">

### Weight Update formula
<img src="https://render.githubusercontent.com/render/math?math=w = w - \eta\frac{\partial y}{\partial x}">

### Applying Chain rule to calculate **new weight w2**
<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial e}{\partial w_{2}} = \frac{\partial e}{\partial a_{2}}.\frac{\partial a_{2}}{\partial z_{2}}.\frac{\partial z_{2}}{\partial w_{2}}">

### Cost function for the NN as a function of w1
<img src="https://render.githubusercontent.com/render/math?math=a_{1} = \sigma (z_{1}) \rightarrow f(z_{1})">
<img src="https://render.githubusercontent.com/render/math?math=z_{1} = w_{1}a_{0} \rightarrow f(w_{1}, a_{0})">

### Weight Update formula
<img src="https://render.githubusercontent.com/render/math?math=w = w - \eta\frac{\partial y}{\partial x}">

### Applying Chain rule to calculate **new weight w1**
<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial e}{\partial w_{1}} = \frac{\partial e}{\partial a_{2}}.\frac{\partial a_{2}}{\partial z_{2}}.\frac{\partial z_{2}}{\partial a_{1}}.\frac{\partial a_{1}}{\partial z_{1}}.\frac{\partial z_{1}}{\partial w_{1}}">

* With same approach we can find updated **Biases** as well
* In the end, we would have a weight metrix and a bias matrix for the whole NN
<img src="https://render.githubusercontent.com/render/math?math=\begin{bmatrix}w1\\ w2\\ ..\\wn\end{bmatrix}">
<img src="https://render.githubusercontent.com/render/math?math=\begin{bmatrix}b1\\ b2\\ ..\\bn\end{bmatrix}">

### Gradient Descent
* The derivative term in above formula is the Gradient Descent
* As Derivative/Gradient (change in value of y with **small** change in value of x) gives direction  of **ASCENT**, hence in order to minimise loss we move in opposite direction **(Descent)** therefore we reduce the original weight (negative sign in above formula)
* Different Optimizers can be tried to find this **Gradient** efficiently

#### Weights Initialised on negative side
* Learning rate is +ve, Gradient is -ve, 
* Product of Learning rate and Gradient will be **NEGATIVE** (slope/tangent/Tan Theta > 90 degrees is a **negative** number)
* As we have Initialised weights on -ve side, so as per weight update formula we would increase the **NEGATIVE** number
* Which would eventually mean, moving towards the **ZERO**

<img src="https://github.com/sbhrwl/dl_experiments/blob/main/artifacts/images/WeightInitilaisedOnNegativeSide.png" width="400"/>

### Types of optimisers
refer [notebook](https://colab.research.google.com/drive/1FEYtjPH5GBY0KVUrX7bFaxRy736WLQvz?usp=sharing)

### MlFlow experiments with Optimizers
<img src='https://drive.google.com/uc?id=1EBeQ3Ec-xPW98yiSqa_-qrvpXJgLIkHO'>

* Run 3: batch_size: **100**, epoch 20, Activation function: relu, Optimizer: SGD - Accuracy: 0.959
* Run 4: batch_size: 100, epoch 20, Activation function: relu, Optimizer: **Adam** - Accuracy: **0.985**
* Run 5: batch_size: 100, epoch 20, Activation function: relu, Optimizer: Adam, **batch normalisation** - Accuracy: **0.994**
* Run 6: batch_size: 100, epoch 20, Activation function: relu, Optimizer: Adam, batch normalisation with **Bias as false** - Accuracy: 0.993
* Run 7: batch_size: 100, epoch 20, Activation function: **sigmoid**, Optimizer: SGD - Accuracy: 0.922
* Run 8: batch_size: 100, epoch 20, Activation function: sigmoid, Optimizer: **Adam** - Accuracy: **0.997**

## Regularisation Techniques
### L1 Regularisation
* LASSO (Least Absolute Shrinkage and Selection Operator) Regularisation
* It is L1 vector norm of vector **w**
* |w| is differentiable everywhere except when w=0
<img src="https://github.com/sbhrwl/dl_experiments/blob/main/artifacts/images/L1-representation.png" width="150"/>

* Equation
  * Alpha is L1 regularisation term
<img src="https://render.githubusercontent.com/render/math?math=J_{n}(\theta) = J_{0}(\theta) %2B \alpha \sum_{i=1}^{m}|\theta|">

## MlFlow experiments with Regularisation techniques 
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

## Transfer Learning
* Transfer learning enables us to use the already built robust models
* Steps for Transfer learning
  * Step 1: Load Previous model
  * Step 2: Check Model Details (available Trainable layers)
  * Step 3: Remove Last Layer
  * Step 4: Create new model
  ```python
    lower_pretrained_layers = model.layers[:-1]

    new_model = tf.keras.models.Sequential(lower_pretrained_layers)
    if is_even == "yes":
        new_model.add(tf.keras.layers.Dense(2, activation=new_layer_activation, name=new_layer_name))
    else:
        new_model.add(tf.keras.layers.Dense(50, activation=new_layer_activation, name=new_layer_name))
        new_model.add(tf.keras.layers.Dense(10, activation=new_output_layer_activation, name=new_output_layer_name))
  ```
* Train the new Model which now already has weights for older layers
```
Layer (type)                 Output Shape              Param #
=================================================================
InputLayer (Flatten)         (None, 784)               0
_________________________________________________________________
hidden_layer_1 (Dense)       (None, 300)               235500
_________________________________________________________________
hidden_layer_2 (Dense)       (None, 100)               30100
_________________________________________________________________
hidden_layer_3 (Dense)       (None, 50)                5050
_________________________________________________________________
new_output_layer (Dense)     (None, 10)                510
=================================================================
Total params: 271,160
Trainable params: 5,560
Non-trainable params: 265,600
_________________________________________________________________
```
* Trainable parameters are only from the new layer plus new output layer (5560)
```python
python src/transfer_learning/ann_mnist_transfer_learning.py
```

### Use Case: Is the Number even?
* We will train it to get weights for new layer(s) added to the model to suit our use case (example: Is the numer Even?)
```python
python src/transfer_learning/is_even.py
```

* [Latex](https://latex.codecogs.com/eqneditor/editor.php)
* [Netron](https://netron.app/)
* [Sketch](https://app.sketchup.com/app?hl=en#)
* [3D plots](https://c3d.libretexts.org/CalcPlot3D/index.html)
