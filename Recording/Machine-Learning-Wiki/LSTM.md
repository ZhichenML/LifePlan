# Long Short-term Memory Unit
## Mechanism

![](https://github.com/ZhichenML/LifePlan/blob/master/Recording/Machine-Learning-Wiki/fig/LSTM.png)

Long short-term memory (LSTM) network is an appealing and powerful Recurrent neural network (RNN) paradigm. LSTM mitigates the gradient vanishing and exploding problems when training RNNs. This is achieved by replacing the original nonlinear hidden units with a linear cell unit, and additional three gates, i.e. the forget gate, the input gate and the output gate, which are parallel filters.

The linear cell state is the main information flow.The purple line in the above figure illustrates the forget gate. It decides how much information should be forget from the last cell state. The sigmoid activation function is used here because the output range is [0,1], which scales the information.
<!--f_{t}=\sigma (W_{f}[h(t-1),x(t)] +b_{f})-->
<a href="https://www.codecogs.com/eqnedit.php?latex=f_{t}=\sigma&space;(W_{f}[h(t-1),x(t)]&space;&plus;b_{f})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f_{t}=\sigma&space;(W_{f}[h(t-1),x(t)]&space;&plus;b_{f})" title="f_{t}=\sigma (W_{f}[h(t-1),x(t)] +b_{f})" /></a>

The blue line demonstrates the input gate. It decides how much information from the cadidate cell state should be passed to the formulate the new cell state. The candidate cell state can be understood as the state of a typical RNN.
<!--i_{t}=\sigma (W_{i}[h(t-1),x(t)] +b_{i})-->
<a href="https://www.codecogs.com/eqnedit.php?latex=i_{t}=\sigma&space;(W_{i}[h(t-1),x(t)]&space;&plus;b_{i})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i_{t}=\sigma&space;(W_{i}[h(t-1),x(t)]&space;&plus;b_{i})" title="i_{t}=\sigma (W_{i}[h(t-1),x(t)] +b_{i})" /></a>
<!--g(t)=\widetilde{s}_{t}=tanh (W_{g}[h(t-1),x(t)] +b_{g})-->
<a href="https://www.codecogs.com/eqnedit.php?latex=g(t)=\widetilde{s}_{t}=tanh&space;(W_{g}[h(t-1),x(t)]&space;&plus;b_{g})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g(t)=\widetilde{s}_{t}=tanh&space;(W_{g}[h(t-1),x(t)]&space;&plus;b_{g})" title="g(t)=\widetilde{s}_{t}=tanh (W_{g}[h(t-1),x(t)] +b_{g})" /></a>

The new cell state is updated as:
<!--s_{t}=f(t)\times s(t-1)+i(t) \times g(t)-->
<a href="https://www.codecogs.com/eqnedit.php?latex=s_{t}=f(t)\times&space;s(t-1)&plus;i(t)&space;\times&space;g(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s_{t}=f(t)\times&space;s(t-1)&plus;i(t)&space;\times&space;g(t)" title="s_{t}=f(t)\times s(t-1)+i(t) \times g(t)" /></a>

The ouput gate decides what information should be passed to the next step to provide guidance about what should be noted. The output gate is demonstrated by the red lines.
<!--o(t)=\sigma(W_{o}[h(t-1),x(t)]+b_{o})-->
<a href="https://www.codecogs.com/eqnedit.php?latex=o(t)=\sigma(W_{o}[h(t-1),x(t)]&plus;b_{o})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?o(t)=\sigma(W_{o}[h(t-1),x(t)]&plus;b_{o})" title="o(t)=\sigma(W_{o}[h(t-1),x(t)]+b_{o})" /></a>
<!--h(t) = o_{t} \times tanh(s(t))-->
<a href="https://www.codecogs.com/eqnedit.php?latex=h(t)&space;=&space;o_{t}&space;\times&space;tanh(s(t))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h(t)&space;=&space;o_{t}&space;\times&space;tanh(s(t))" title="h(t) = o_{t} \times tanh(s(t))" /></a>

It is apparent that the sigmoid activation function is employed as a scaling function for the main cell state. The tanh activation function is employed to transform the original value into symmetric information values that could be easily scaled by the sigmoid outputs. 

Having found this, we could omit the scaling components from the above figure. 
That lead to a simplified explanation: 
1. The current input 
<a href="https://www.codecogs.com/eqnedit.php?latex=x(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x(t)" title="x(t)" /></a>
and the last hidden state <a href="https://www.codecogs.com/eqnedit.php?latex=h(t-1)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h(t-1)" title="h(t-1)" /></a>
formulate an update term <a href="https://www.codecogs.com/eqnedit.php?latex=g(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g(t)" title="g(t)" /></a>.  
2. the last cell state is added by <a href="https://www.codecogs.com/eqnedit.php?latex=g(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g(t)" title="g(t)" /></a>, resulting in a new cell state.
3. The new cell state is rescaled by <a href="https://www.codecogs.com/eqnedit.php?latex=o(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?o(t)" title="o(t)" /></a> to formulate a new hidden state.

Then, we add the forget gate and the input gate to rescale the last cell state and the update term respectively (In this sense, the input gate and the forget gate could be determined by the same thing, which leads to Gated recurrent units). And we add the output gate to rescale the new cell state to preserve information to the next step.


## Gated Recurrent Unit (GRU)
GRU is a simplified LSTM. GRU uses the hidden state to replace the cell state and the hidden state in the LSTM. It also combines the forget gate and input gate so that they sum to be 1. Thereofore, only one value need to be determined, as shown in the follwing figure.

![](https://github.com/ZhichenML/LifePlan/blob/master/Recording/Machine-Learning-Wiki/fig/GRU.png)

<!--r_{t}=\sigma(W_{r}[h_{t-1},x_{t}])-->
<a href="https://www.codecogs.com/eqnedit.php?latex=r_{t}=\sigma(W_{r}[h_{t-1},x_{t}])" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r_{t}=\sigma(W_{r}[h_{t-1},x_{t}])" title="r_{t}=\sigma(W_{r}[h_{t-1},x_{t}])" /></a>
<!--z_{t}=\sigma(W_{z}[h_{t-1},x_{t}])-->
<a href="https://www.codecogs.com/eqnedit.php?latex=z_{t}=\sigma(W_{z}[h_{t-1},x_{t}])" target="_blank"><img src="https://latex.codecogs.com/gif.latex?z_{t}=\sigma(W_{z}[h_{t-1},x_{t}])" title="z_{t}=\sigma(W_{z}[h_{t-1},x_{t}])" /></a>
<!--\widetilde{h_{t}}= tanh(W_{h}[r_{t} \times h_{t-1}, x_{t}]])-->
<a href="https://www.codecogs.com/eqnedit.php?latex=\widetilde{h_{t}}=&space;tanh(W_{h}[r_{t}&space;\times&space;h_{t-1},&space;x_{t}]])" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\widetilde{h_{t}}=&space;tanh(W_{h}[r_{t}&space;\times&space;h_{t-1},&space;x_{t}]])" title="\widetilde{h_{t}}= tanh(W_{h}[r_{t} \times h_{t-1}, x_{t}]])" /></a>
<!--h_{t}=z_{t} \times \widetilde{h_{t}} + (1-z_{t})\times h_{t-1} )-->
<a href="https://www.codecogs.com/eqnedit.php?latex=h_{t}=z_{t}&space;\times&space;\widetilde{h_{t}}&space;&plus;&space;(1-z_{t})\times&space;h_{t-1}&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_{t}=z_{t}&space;\times&space;\widetilde{h_{t}}&space;&plus;&space;(1-z_{t})\times&space;h_{t-1}&space;)" title="h_{t}=z_{t} \times \widetilde{h_{t}} + (1-z_{t})\times h_{t-1} )" /></a>

# Training

It's relatively easy to understand the mechanism of LSTM's forward pass. That is, the sequential samples arrives one by one and activate the corresponding hidden states, with the fixed network parameters.

Now, after the last time point of the input sequence arrived, we begin to propagate the error signals backwards to tune the network parameters.

First, let us assume the LSTM prediction for each time point is compared to the target output sequences by Euclidean loss (using the first dimension of the hidden state as the output),
<!--l(t) = \frac{1}{2}(y(t)-h(t))^2-->
<a href="https://www.codecogs.com/eqnedit.php?latex=l(t)&space;=&space;\frac{1}{2}(y(t)-h(t))^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l(t)&space;=&space;\frac{1}{2}(y(t)-h(t))^2" title="l(t) = \frac{1}{2}(y(t)-h(t))^2" /></a>

The overall loss is 
<!--L = \sum^{T}_{t=1}l(t)-->
<a href="https://www.codecogs.com/eqnedit.php?latex=L&space;=&space;\sum^{T}_{t=1}l(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L&space;=&space;\sum^{T}_{t=1}l(t)" title="L = \sum^{T}_{t=1}l(t)" /></a>

The activations of the hidden states:
<!--\\
g(t) = tanh(W_g[h(t-1),x(t)]+b_{g})\\
i(t) = \sigma(W_{i}[h(t-1),x(t)]+b_{i})\\
f(t) = \sigma(W_{f}[h(t-1),x(t)]+b_{f})\\
o(t) = \sigma(W_{o}[h(t-1),x(t)]+b_{o})\\
s(t) = f(t) \times s(t-1) + i(t) \times g(t) \\
h(t) = s(t) \times o(t)-->
<a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;g(t)&space;=&space;tanh(W_g[h(t-1),x(t)]&plus;b_{g})\\&space;i(t)&space;=&space;\sigma(W_{i}[h(t-1),x(t)]&plus;b_{i})\\&space;f(t)&space;=&space;\sigma(W_{f}[h(t-1),x(t)]&plus;b_{f})\\&space;o(t)&space;=&space;\sigma(W_{o}[h(t-1),x(t)]&plus;b_{o})\\&space;s(t)&space;=&space;f(t)&space;\times&space;s(t-1)&space;&plus;&space;i(t)&space;\times&space;g(t)&space;\\&space;h(t)&space;=&space;s(t)&space;\times&space;o(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;g(t)&space;=&space;tanh(W_g[h(t-1),x(t)]&plus;b_{g})\\&space;i(t)&space;=&space;\sigma(W_{i}[h(t-1),x(t)]&plus;b_{i})\\&space;f(t)&space;=&space;\sigma(W_{f}[h(t-1),x(t)]&plus;b_{f})\\&space;o(t)&space;=&space;\sigma(W_{o}[h(t-1),x(t)]&plus;b_{o})\\&space;s(t)&space;=&space;f(t)&space;\times&space;s(t-1)&space;&plus;&space;i(t)&space;\times&space;g(t)&space;\\&space;h(t)&space;=&space;s(t)&space;\times&space;o(t)" title="\\ g(t) = tanh(W_g[h(t-1),x(t)]+b_{g})\\ i(t) = \sigma(W_{i}[h(t-1),x(t)]+b_{i})\\ f(t) = \sigma(W_{f}[h(t-1),x(t)]+b_{f})\\ o(t) = \sigma(W_{o}[h(t-1),x(t)]+b_{o})\\ s(t) = f(t) \times s(t-1) + i(t) \times g(t) \\ h(t) = s(t) \times o(t)" /></a>

As we see, the LSTM cell is actually four parallel feedforward neural networks inside. For the cell's input and output, if we unfold the states along time axis, each time point (a node) is connected to the next time point via <a href="https://www.codecogs.com/eqnedit.php?latex=s(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s(t)" title="s(t)" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=h(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h(t)" title="h(t)" /></a>. Given the activations and error signal flows of these two variables, we are able to perform forward pass and backpropogation through time.

Assume we have only one LSTM cell, then
<!--\\
\frac{dL}{dW}=\sum_{t=1}^{T}\frac{dL}{dh(t)}\frac{dh(t)}{dW}\\
=\sum_{t=1}^{T}\frac{d\sum_{s=1}^{T}l(t)}{dh(t)}\frac{dh(t)}{dW}\\
=\sum_{t=1}^{T}\frac{d\sum_{s=t}^{T}l(t)}{dh(t)}\frac{dh(t)}{dW}\\
=\sum_{t=1}^{T}\frac{dL(t)}{dh(t)}\frac{dh(t)}{dW}-->
<a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;\frac{dL}{dW}=\sum_{t=1}^{T}\frac{dL}{dh(t)}\frac{dh(t)}{dW}\\&space;=\sum_{t=1}^{T}\frac{d\sum_{s=1}^{T}l(t)}{dh(t)}\frac{dh(t)}{dW}\\&space;=\sum_{t=1}^{T}\frac{d\sum_{s=t}^{T}l(t)}{dh(t)}\frac{dh(t)}{dW}\\&space;=\sum_{t=1}^{T}\frac{dL(t)}{dh(t)}\frac{dh(t)}{dW}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;\frac{dL}{dW}=\sum_{t=1}^{T}\frac{dL}{dh(t)}\frac{dh(t)}{dW}\\&space;=\sum_{t=1}^{T}\frac{d\sum_{s=1}^{T}l(t)}{dh(t)}\frac{dh(t)}{dW}\\&space;=\sum_{t=1}^{T}\frac{d\sum_{s=t}^{T}l(t)}{dh(t)}\frac{dh(t)}{dW}\\&space;=\sum_{t=1}^{T}\frac{dL(t)}{dh(t)}\frac{dh(t)}{dW}" title="\\ \frac{dL}{dW}=\sum_{t=1}^{T}\frac{dL}{dh(t)}\frac{dh(t)}{dW}\\ =\sum_{t=1}^{T}\frac{d\sum_{s=1}^{T}l(t)}{dh(t)}\frac{dh(t)}{dW}\\ =\sum_{t=1}^{T}\frac{d\sum_{s=t}^{T}l(t)}{dh(t)}\frac{dh(t)}{dW}\\ =\sum_{t=1}^{T}\frac{dL(t)}{dh(t)}\frac{dh(t)}{dW}" /></a>

where 
<!--L(t) = \sum_{s=t}^{T}l(t)--> 
<a href="https://www.codecogs.com/eqnedit.php?latex=L(t)&space;=&space;\sum_{s=t}^{T}l(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(t)&space;=&space;\sum_{s=t}^{T}l(t)" title="L(t) = \sum_{s=t}^{T}l(t)" /></a>

The first term in the summation <!--\frac{dL(t)}{dh(t)}--> <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{dL(t)}{dh(t)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{dL(t)}{dh(t)}" title="\frac{dL(t)}{dh(t)}" /></a> is the error signal flowing between nodes through time. We have 

<!--\\
\frac{dL(t)}{dh(t)}\\
=\frac{dl(t)}{dh(t)} + \frac{dL(t+1)}{dh(t)}\\\\-->
<a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;\frac{dL(t)}{dh(t)}\\&space;=\frac{dl(t)}{dh(t)}&space;&plus;&space;\frac{dL(t&plus;1)}{dh(t)}\\\\" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;\frac{dL(t)}{dh(t)}\\&space;=\frac{dl(t)}{dh(t)}&space;&plus;&space;\frac{dL(t&plus;1)}{dh(t)}\\\\" title="\\ \frac{dL(t)}{dh(t)}\\ =\frac{dl(t)}{dh(t)} + \frac{dL(t+1)}{dh(t)}\\\\" /></a>

where the first term represents the loss of the current output, the second term represents the the loss of the remaining time points. 

The cell state <!--s(t)--><a href="https://www.codecogs.com/eqnedit.php?latex=s(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s(t)" title="s(t)" /></a> controls the other error signal flowing between nodes.
<!--\\
\frac{dL(t)}{ds(t)} = \frac{dL(t)}{dh(t)}\frac{dh(t)}{ds(t)} + \frac{dL(t)}{dh(t+1)}\frac{dh(t+1)}{ds(t)}\\
=\frac{dL(t)}{dh(t)}\frac{dh(t)}{ds(t)} + \frac{dL(t+1)}{dh(t+1)}\frac{dh(t+1)}{ds(t)}\\
=\frac{dL(t)}{dh(t)}\frac{dh(t)}{ds(t)} + \frac{dL(t+1)}{ds(t)}\\
=\frac{dL(t)}{dh(t)}o(t) + \frac{dL(t+1)}{ds(t)}\\-->

<a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;\frac{dL(t)}{ds(t)}&space;=&space;\frac{dL(t)}{dh(t)}\frac{dh(t)}{ds(t)}&space;&plus;&space;\frac{dL(t)}{dh(t&plus;1)}\frac{dh(t&plus;1)}{ds(t)}\\&space;=\frac{dL(t)}{dh(t)}\frac{dh(t)}{ds(t)}&space;&plus;&space;\frac{dL(t&plus;1)}{dh(t&plus;1)}\frac{dh(t&plus;1)}{ds(t)}\\&space;=\frac{dL(t)}{dh(t)}\frac{dh(t)}{ds(t)}&space;&plus;&space;\frac{dL(t&plus;1)}{ds(t)}\\&space;=\frac{dL(t)}{dh(t)}o(t)&space;&plus;&space;\frac{dL(t&plus;1)}{ds(t)}\\" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;\frac{dL(t)}{ds(t)}&space;=&space;\frac{dL(t)}{dh(t)}\frac{dh(t)}{ds(t)}&space;&plus;&space;\frac{dL(t)}{dh(t&plus;1)}\frac{dh(t&plus;1)}{ds(t)}\\&space;=\frac{dL(t)}{dh(t)}\frac{dh(t)}{ds(t)}&space;&plus;&space;\frac{dL(t&plus;1)}{dh(t&plus;1)}\frac{dh(t&plus;1)}{ds(t)}\\&space;=\frac{dL(t)}{dh(t)}\frac{dh(t)}{ds(t)}&space;&plus;&space;\frac{dL(t&plus;1)}{ds(t)}\\&space;=\frac{dL(t)}{dh(t)}o(t)&space;&plus;&space;\frac{dL(t&plus;1)}{ds(t)}\\" title="\\ \frac{dL(t)}{ds(t)} = \frac{dL(t)}{dh(t)}\frac{dh(t)}{ds(t)} + \frac{dL(t)}{dh(t+1)}\frac{dh(t+1)}{ds(t)}\\ =\frac{dL(t)}{dh(t)}\frac{dh(t)}{ds(t)} + \frac{dL(t+1)}{dh(t+1)}\frac{dh(t+1)}{ds(t)}\\ =\frac{dL(t)}{dh(t)}\frac{dh(t)}{ds(t)} + \frac{dL(t+1)}{ds(t)}\\ =\frac{dL(t)}{dh(t)}o(t) + \frac{dL(t+1)}{ds(t)}\\" /></a>

where the first row is because <!--s(t)--> <a href="https://www.codecogs.com/eqnedit.php?latex=s(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s(t)" title="s(t)" /></a> is devided into two parts in LSTM. One is to generate the hidden state <!--h(t)--> <a href="https://www.codecogs.com/eqnedit.php?latex=h(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h(t)" title="h(t)" /></a>. The other is to flow into the next time node. So the result is not surprising.

Having understand the connecting error signals flowing between the time nodes, now we turn to the error propagation inside the LSTM time nodes. In this case, we assume we have got <!--\frac{dL(t)}{dh(t)}--> <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{dL(t)}{dh(t)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{dL(t)}{dh(t)}" title="\frac{dL(t)}{dh(t)}" /></a> and <!--\frac{dL(t)}{ds(t)}--> <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{dL(t)}{ds(t)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{dL(t)}{ds(t)}" title="\frac{dL(t)}{ds(t)}" /></a> (This is achieved by backpropagation from the last time nodes, since <!--\frac{dL(T)}{dh(T)}=\frac{dl(T)}{dh(T)}--> <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{dL(T)}{dh(T)}=\frac{dl(T)}{dh(T)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{dL(T)}{dh(T)}=\frac{dl(T)}{dh(T)}" title="\frac{dL(T)}{dh(T)}=\frac{dl(T)}{dh(T)}" /></a> and <!--\frac{dL(T)}{ds(T)}=\frac{dl(T)}{ds(T)}=\frac{dl(T)}{dh(T)} \times o(t)--> <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{dL(T)}{ds(T)}=\frac{dl(T)}{ds(T)}=\frac{dl(T)}{dh(T)}&space;\times&space;o(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{dL(T)}{ds(T)}=\frac{dl(T)}{ds(T)}=\frac{dl(T)}{dh(T)}&space;\times&space;o(t)" title="\frac{dL(T)}{ds(T)}=\frac{dl(T)}{ds(T)}=\frac{dl(T)}{dh(T)} \times o(t)" /></a>).
Then in the time node for time point <a href="https://www.codecogs.com/eqnedit.php?latex=t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?t" title="t" /></a>,

<!--\\
\frac{dL(t)}{dg(t)}=\frac{dL(t)}{ds(t)}i(t)\\
\frac{dL(t)}{di(t)}=\frac{dL(t)}{ds(t)}g(t)\\
\frac{dL(t)}{df(t)}=\frac{dL(t)}{ds(t)}s(t-1)\\
\frac{dL(t)}{do(t)}=\frac{dL(t)}{dh(t)}s(t)\\-->
<a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;\frac{dL(t)}{dg(t)}=\frac{dL(t)}{ds(t)}i(t)\\&space;\frac{dL(t)}{di(t)}=\frac{dL(t)}{ds(t)}g(t)\\&space;\frac{dL(t)}{df(t)}=\frac{dL(t)}{ds(t)}s(t-1)\\&space;\frac{dL(t)}{do(t)}=\frac{dL(t)}{dh(t)}s(t)\\" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;\frac{dL(t)}{dg(t)}=\frac{dL(t)}{ds(t)}i(t)\\&space;\frac{dL(t)}{di(t)}=\frac{dL(t)}{ds(t)}g(t)\\&space;\frac{dL(t)}{df(t)}=\frac{dL(t)}{ds(t)}s(t-1)\\&space;\frac{dL(t)}{do(t)}=\frac{dL(t)}{dh(t)}s(t)\\" title="\\ \frac{dL(t)}{dg(t)}=\frac{dL(t)}{ds(t)}i(t)\\ \frac{dL(t)}{di(t)}=\frac{dL(t)}{ds(t)}g(t)\\ \frac{dL(t)}{df(t)}=\frac{dL(t)}{ds(t)}s(t-1)\\ \frac{dL(t)}{do(t)}=\frac{dL(t)}{dh(t)}s(t)\\" /></a>

The next step is to go inside each gate for the derivative of the network parameters,
<!--\\
\frac{dL(t)}{dW_g} = [x(t),h(t-1)]^T \otimes \frac{dL(t)}{dg(t)}(1-g(t)^2) \\
\frac{dL(t)}{dW_i} = [x(t),h(t-1)]^T \otimes \frac{dL(t)}{di(t}i(t)(1-i(t))\\
\frac{dL(t)}{dW_f} = [x(t),h(t-1)]^T \otimes \frac{dL(t)}{df(t}f(t)(1-f(t))\\
\frac{dL(t)}{dW_o} = [x(t),h(t-1)]^T \otimes \frac{dL(t)}{do(t}o(t)(1-o(t))-->
<a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;\frac{dL(t)}{dW_g}&space;=&space;[x(t),h(t-1)]^T&space;\otimes&space;\frac{dL(t)}{dg(t)}(1-g(t)^2)&space;\\&space;\frac{dL(t)}{dW_i}&space;=&space;[x(t),h(t-1)]^T&space;\otimes&space;\frac{dL(t)}{di(t}i(t)(1-i(t))\\&space;\frac{dL(t)}{dW_f}&space;=&space;[x(t),h(t-1)]^T&space;\otimes&space;\frac{dL(t)}{df(t}f(t)(1-f(t))\\&space;\frac{dL(t)}{dW_o}&space;=&space;[x(t),h(t-1)]^T&space;\otimes&space;\frac{dL(t)}{do(t}o(t)(1-o(t))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;\frac{dL(t)}{dW_g}&space;=&space;[x(t),h(t-1)]^T&space;\otimes&space;\frac{dL(t)}{dg(t)}(1-g(t)^2)&space;\\&space;\frac{dL(t)}{dW_i}&space;=&space;[x(t),h(t-1)]^T&space;\otimes&space;\frac{dL(t)}{di(t}i(t)(1-i(t))\\&space;\frac{dL(t)}{dW_f}&space;=&space;[x(t),h(t-1)]^T&space;\otimes&space;\frac{dL(t)}{df(t}f(t)(1-f(t))\\&space;\frac{dL(t)}{dW_o}&space;=&space;[x(t),h(t-1)]^T&space;\otimes&space;\frac{dL(t)}{do(t}o(t)(1-o(t))" title="\\ \frac{dL(t)}{dW_g} = [x(t),h(t-1)]^T \otimes \frac{dL(t)}{dg(t)}(1-g(t)^2) \\ \frac{dL(t)}{dW_i} = [x(t),h(t-1)]^T \otimes \frac{dL(t)}{di(t}i(t)(1-i(t))\\ \frac{dL(t)}{dW_f} = [x(t),h(t-1)]^T \otimes \frac{dL(t)}{df(t}f(t)(1-f(t))\\ \frac{dL(t)}{dW_o} = [x(t),h(t-1)]^T \otimes \frac{dL(t)}{do(t}o(t)(1-o(t))" /></a>

It means the error signal go through the activation functions, and perform outer product with inputs of the time nodes (for intermediate nodes, it is <!--[h(t-1),x(t)]--> <a href="https://www.codecogs.com/eqnedit.php?latex=[h(t-1),x(t)]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?[h(t-1),x(t)]" title="[h(t-1),x(t)]" /></a> and for the first node of the first time point, it is <!--[\overrightarrow{0},x(t)]--> <a href="https://www.codecogs.com/eqnedit.php?latex=[\overrightarrow{0},x(t)]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?[\overrightarrow{0},x(t)]" title="[\overrightarrow{0},x(t)]" /></a>). 

And for the bias terms,

<!--\\
\frac{dL(t)}{db_g} = \frac{dL(t)}{dg(t)}(1-g(t)^2) \\
\frac{dL(t)}{db_i} = \frac{dL(t)}{di(t}i(t)(1-i(t))\\
\frac{dL(t)}{db_f} = \frac{dL(t)}{df(t}f(t)(1-f(t))\\
\frac{dL(t)}{db_o} = \frac{dL(t)}{do(t}o(t)(1-o(t))-->
<a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;\frac{dL(t)}{db_g}&space;=&space;\frac{dL(t)}{dg(t)}(1-g(t)^2)&space;\\&space;\frac{dL(t)}{db_i}&space;=&space;\frac{dL(t)}{di(t}i(t)(1-i(t))\\&space;\frac{dL(t)}{db_f}&space;=&space;\frac{dL(t)}{df(t}f(t)(1-f(t))\\&space;\frac{dL(t)}{db_o}&space;=&space;\frac{dL(t)}{do(t}o(t)(1-o(t))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;\frac{dL(t)}{db_g}&space;=&space;\frac{dL(t)}{dg(t)}(1-g(t)^2)&space;\\&space;\frac{dL(t)}{db_i}&space;=&space;\frac{dL(t)}{di(t}i(t)(1-i(t))\\&space;\frac{dL(t)}{db_f}&space;=&space;\frac{dL(t)}{df(t}f(t)(1-f(t))\\&space;\frac{dL(t)}{db_o}&space;=&space;\frac{dL(t)}{do(t}o(t)(1-o(t))" title="\\ \frac{dL(t)}{db_g} = \frac{dL(t)}{dg(t)}(1-g(t)^2) \\ \frac{dL(t)}{db_i} = \frac{dL(t)}{di(t}i(t)(1-i(t))\\ \frac{dL(t)}{db_f} = \frac{dL(t)}{df(t}f(t)(1-f(t))\\ \frac{dL(t)}{db_o} = \frac{dL(t)}{do(t}o(t)(1-o(t))" /></a>

This is because the derivative for the bias term <a href="https://www.codecogs.com/eqnedit.php?latex=b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?b" title="b" /></a> is 1 inside the gate, so it is just the error signal before the activation.

Having got the derivative for the cell networks, we have to continue calculate the derivative for the inputs of the time node, i.e. <a href="https://www.codecogs.com/eqnedit.php?latex=s(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?s(t)" title="s(t)" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=h(t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h(t)" title="h(t)" /></a> for the previous time point. In this way, the error could back propagate from the last time point to the first time point. The derivatives of the network parameters are accumulated through time. When the error propagation arrpaches the first time point, the network parameters are updated by subtracting the accumulated derivatives. If we have an output layer, such as linear regression, the regression parameters are also accumulated through time. When updating, it should be devided by the number of time points. Otherwise, it would be unstable in optimization.

The following figure demonstrates the overall process. The input and output sequence both have 3 time points.

![](https://github.com/ZhichenML/LifePlan/blob/master/Recording/Machine-Learning-Wiki/fig/BPTT.png)





## Reference

> Hochreiter, Sepp, and Jürgen Schmidhuber. "Long short-term memory." Neural computation 9.8 (1997): 1735-1780.

> http://colah.github.io/posts/2015-08-Understanding-LSTMs/

> He, Zhen, et al. "Wider and Deeper, Cheaper and Faster: Tensorized LSTMs for Sequence Learning." Advances in Neural Information Processing Systems. 2017.

> https://nicodjimenez.github.io/2014/08/08/lstm.html

> http://arunmallya.github.io/writeups/nn/lstm/index.html#/

> https://github.com/nicodjimenez/lstm
