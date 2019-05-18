I finally comple this sparse AutoEncoder. The AutoEncoder is a two layer network with the objective to reconstruct the original input.
The objective function include an average square error, a squared-form weight decay term and a sparsity term.
The sparsity term is defined as the KL divergence between the hidden activations and a small value. This means the goal is to make a
node in the hidden layer not easy to be active after seeing an input. As a consequence, only very value information in the input is 
emphasized, while others are conpressed. 

To accomplish this code, the key is the gradient calculation. The strategy is to first implement the function that returns the cost value
and the gradient, then compare with the numerical gradient to varify.
The sequare form can be mainly backpropagated based on residual error, which is the gradient w.r.t. each node's input.
Note that for the sparisty term, it involves the hidden layer activation, and leads to an extra term when calculating the residual error.
Another thing to note is that the gradient is w.r.t. each instance, thus the activation is repeated for #instance times.

Finally the code visiualizes the input layer, i.e. each hidden node's input weights, which is filters.

for matlab,
it is worthy to use more function handle operator @ than functions.

I found an interesting blog: http://www.cnblogs.com/tornadomeet/archive/2012/05/24/2515980.html
