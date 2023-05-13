# EE399-A-HW5

## Neural Networks for Dynamical Systems
Marcus Chen, May 12, 2023

As shown in previous projects, machine learning processes are inherently mathematical and function-based; the same can be said for neural networks. Because of the comparatively more complex anatomy of a neural network, neural networks can be utilized for a diverse range of mathematical tasks. In this project, we will be using multiple neural networks: FFNN’s, RNN’s, LSTM’s, and Echo State Networks (ESN’s), to perform future state predictions on the dynamic Laurenz System of Equations. 

### Introduction:
In previous projects, we used machine learning functions and neural networks to perform classification and regression based on data that was mostly centered around computer vision, but neural networks have a wider variety of mathematical capabilities. 

Instead of using a dataset like previously, in this project, we will be creating our own datasets based on the dynamical system Laurenz System of Equations:

$\dot{x}=\sigma(y-x)$

$\dot{y}=x(\rho-z)-y$

$\dot{z}=xy-\beta z$

where the state of the system is given by:

$\textbf{x}=[x y z]^T$

with the parameters: $\beta$, $\sigma$, and $\rho$

For the project, we will look at varying values for $\rho$.

A three-layer FFNN with the layers:

```
ReLU
ReLU
Linear
```

is trained to perform future state prediction, based on the Lorenz systems with rho values 10, 28, and 40. The FFNN will then be tested with the Lorenz systems rho values of 17 and 35. 

The accuracy of the neural network will then be compared to the accuracy of future state predictions made by other neural networks: an RNN, an LSTM, and an ESN.

### Theoretical Background:
#### Dynamical System:
A system in which a function describes the time dependence of a point in ambient space, such as the Lorenz Equation.
![download](https://github.com/marcuschen001/EE399-A-HW5/assets/66970342/d5675852-0430-4657-bb93-37a4451270ff)

#### Echo State Network (ESN):
A type of RNN that has a sparsely connected hidden layer. The connectivity and weights of hidden neurons are randomly assigned, creating a “reservoir” of neurons also known as a neuronal or dynamic reservoir. Even though the network is nonlinear, the only weights that are modified are the synapses between the reservoir and the output neurons and the quadratic error function can be differentiated into something linear. 

![A-Echo-State-Network-consists-of-three-layers-input-layer-reservoir-layer-and](https://github.com/marcuschen001/EE399-A-HW5/assets/66970342/ac86614f-e20a-44b3-ae92-eefa315994c8)

### Algorithm Interpretation and Development:
In order to simulate the Lorenz Equation, we first need to make a function that takes in three-dimensional data and outputs a series of three-dimensional vectors based on the three Lorenz equations, similar to the vector shown above. The programming is done like so:

```
def lorenz_deriv(x_y_z, t0, sigma=sigma, beta=beta, rho=rho[0]):
    x, y, z = x_y_z
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]
```

Our time series data and other constants are written like so:

```
dt = 0.01
T = 8
t = np.arange(0,T+dt,dt)
beta = 8/3
sigma = 10
rho = [10, 17, 28, 35, 40]
```

In order to make initial condition three-dimensional data, we can make 100 random data points for x, y, and z that ranges from -15 to 15 like so:

```
np.random.seed(123)
x0 = -15 + 30 * np.random.random((100, 3))
```

To create the rest of the data in the time series, we can create arrays like so:

```
x_t = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t)
                  for x0_j in x0])
```

But in our case, we also want to record the time series data for more than one value of $\rho$, so we can loop and and create the results like so:

```
x_t = []
for i in range(5):
  x_t_part = np.asarray([integrate.odeint(lorenz_deriv, x0_j, t, args=(sigma, beta, rho[i]))
                  for x0_j in x0])
  x_t.append(x_t_part)
```

To approximate the input and output data, we can iterate through all the time data to find the input and output values of the Lorenz Equations that we can then use to train and test a future state prediction model. It is done like so:

```
nn_input = np.zeros((100*(len(t)-1),3))
nn_output = np.zeros_like(nn_input)

for j in range(100):
    nn_input[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_t[j,:-1,:]
    nn_output[j*(len(t)-1):(j+1)*(len(t)-1),:] = x_t[j,1:,:]
```

To create a simple feed forward neural network with the aforementioned activation equations, it will look like so:

```
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(in_features=3, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=10)
        self.fc3 = nn.Linear(in_features=10, out_features=3)
        self.m = nn.ReLU()
        
    def forward(self, x):
        x = self.m(self.fc1(x))
        x = self.m(self.fc2(x))
        x = purelin(self.fc3(x))
        return x
```

Training and testing the data based on the two other rho values will look like this:

```
# Create model instance
model = MyModel()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(10):
    optimizer.zero_grad()

    for i, data in enumerate(dataloader_train):
      x, y = data
      outputs = model(x.float())
      loss = criterion(outputs, y.float())
      loss.backward()
      optimizer.step()
      if (i+1) % 500 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, 10, i+1, len(dataloader_train), loss.item()))

with torch.no_grad():
    total = 0.0
    for i, data in enumerate(dataloader_test):
      x, y = data
      outputs = model(x.float())
      MSE = (outputs - y.float())**2
      total += MSE
    print('MSE Error: {}'.format(torch.mean(total)))
```

Creating an RNN neural network is done like so:

```
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(3, 10, 2, batch_first=True)
        self.fc = nn.Linear(10, 3)
        
    def forward(self, x):
        h0 = torch.zeros(2, 10)
        x, hidden = self.rnn(x, h0)
        x = self.fc(x)
        return x
```

Creating an LSTM neural network is done like so:

```
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(3, 10, 2, batch_first=True)
        self.fc = nn.Linear(10, 3)
        
    def forward(self, x):
        h0 = torch.zeros(2, 10)
        c0 = torch.zeros(2, 10)
        x, hidden = self.lstm(x, (h0, c0))
        x = self.fc(x)
        return x
```

Because an ESM isn't one of the layer types included in vanilla Pytorch, we are using a version called auto ESM, from this link:
https://pypi.org/project/auto-esn/

We first import it and can use it in the Jupyter notebook as well.

```
pip install auto_esn
```

```
from auto_esn.esn.esn import GroupedDeepESN
import torch.nn as nn
import torch.optim as optim

esn = GroupedDeepESN(input_size=3, groups=4, num_layers=(2, 2), hidden_size=10, output_dim=3)
```

To graph the data, the calculated outputs can also be recorded while the model is tested by just having a way to concatenate the outputs onto an array. Then, outputs can be graphed via a method like so:
```
def graph(array, color):
  for j in range(100):
    x_t_current[j,1:,:] = array[j*(len(t)-1):(j+1)*(len(t)-1),:]
    x, y, z = x_t_current[j,:,:].T
    ax.plot(x, y, z,linewidth=1, color=color)
```

where color can be decided by the user via a character.

### Computational Results:
Before the machine learning model was trained, data was recorded for the Lorenz System across 5 different $\rho$ values:
![download (1)](https://github.com/marcuschen001/EE399-A-HW5/assets/66970342/0d9d6dcd-8089-49a1-81bf-d63a9b6c4525)

All data is taken into one 2D array, with it then being split between training and testing data, where each $\rho$ variant has 80,000 3-dimensional datapoints:

```
X_train = np.concatenate((nn_input[0:80000], nn_input[80000 * 2: 80000 * 3], nn_input[80000 * 4: 80000 * 5]))
X_test = np.concatenate((nn_input[80000:80000 * 2], nn_input[80000 * 3:80000 * 4]))

Y_train = np.concatenate((nn_output[0:80000], nn_output[80000 * 2: 80000 * 3], nn_output[80000 * 4: 80000 * 5]))
Y_test = np.concatenate((nn_output[80000:80000 * 2], nn_output[80000 * 3:80000 * 4]))
```
that were then converted into Tensor datasets and put into a dataloader. 

Using the first FFNN, the results were:
![download (2)](https://github.com/marcuschen001/EE399-A-HW5/assets/66970342/eca87c19-65d6-40b6-b147-7946f688402f)

Where the red is the expectations and the blue are the tested results.

The MSE error is:
```
MSE Error: 117750.7890625
```

While the MSE has been shown to decrease drastically over more epochs, due to time constraints, only 10 epochs were taken for each NN. 

Compared to the RNN and the LSTM, the results were:
![download (3)](https://github.com/marcuschen001/EE399-A-HW5/assets/66970342/fcd8c496-3cf0-4870-bca9-373aed0a28b1)

Where the red is the expectations, the blue is the RNN, and the green is the LSTM.

The MSE error for the RNN is:

```
MSE Error: 82452.671875
```

and the MSE error for the LSTM is:

```
MSE Error: 45929.08203125
```

both significantly better values. 

As for the ESM, there were issues:
![download (4)](https://github.com/marcuschen001/EE399-A-HW5/assets/66970342/794b3ff5-afc2-4650-9447-5ecb1401240c)

The MSE error for the ESM is:

```
MSE Error: 26147386.0
```
There were many possible reasons for the possible issues, from data overload (there WAS over 400,000 datapoints to analyze) or from not fully understanding auto ESN, but due to time constraints, those can be explored later.

Overall, the LSTM did the greatest with the data, decreasing the MSE error of the FFNN by factor of over 2! The RNN also performed better than the standard FFNN, but the ESM did worse.

### Conclusions:

In this project, neural network models were extended upon from previous usages, from simple image processing and linear regression, to simulating basic dynamical systems. A short comparison between different neural networks were also present too: RNN's generally performed better on the dynamical system data than the FFNN, possibly due to the element of time being involved. Coincidentally, the results shown here have also become a trend in real life, with LSTM's being viewed generally more favorably than a one-directional neural network like an FFNN, a slow neural network like an RNN, and an unstable network like an ESN. 

Additionally, while ESN's were generally seen as an amendum to the standard formula of the RNN, it has phased out to the point of not being part of the types of layers one can choose in the Pytorch library. 
