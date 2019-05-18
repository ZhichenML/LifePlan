## Natural gradient
The natural gradient is the standard gradient warped by the Fisher Information metric.

Image optimizing a function <img src="https://latex.codecogs.com/gif.latex?L(\theta)" title="L(\theta)" /> by gradient descent, 

<img src="https://latex.codecogs.com/gif.latex?\theta^{i&plus;1}&space;=&space;\theta^{i}&plus;&space;\lambda&space;L'(\theta)" title="\theta^{i+1} = \theta^{i}+ \lambda L'(\theta)" />

The derivative finds a direction that makes the function descent (for minimization) the most (even this is actually not true).
It assumes updating the parameter with Euclidean distance, that is the scale of each dimension is the same.

However, this assumption may not be true when the parameter space is on a Riemannian manifold, 
such as the parameters of the probability distribution, which is the key idea for variational methods.
For example, N(0,1) and N(1,1) distance = 1, N(0,100) and N(1,100) distance = 1. 
However, the latter pair is almost the same distribution.

To solve this problem, we assume an update <img src="https://latex.codecogs.com/gif.latex?d\phi" title="d\phi" />

<img src="https://latex.codecogs.com/gif.latex?L(\theta&space;&plus;&space;d\phi)&space;=&space;L(\theta)&plus;L'(\theta)d\phi&space;=&space;L(\theta)&space;&plus;&space;L'(\theta)\varepsilon&space;v" title="L(\theta + d\phi) = L(\theta)+L'(\theta)d\phi = L(\theta) + L'(\theta)\varepsilon v" />

and set <img src="https://latex.codecogs.com/gif.latex?v&space;^T&space;G(\phi)v&space;=&space;1" title="v ^T G(\phi)v = 1" />, meaning
the update is unitary under the geometry.

By using Lagrange multiplier, we have 

<img src="https://latex.codecogs.com/gif.latex?L(\theta)&space;&plus;&space;L'(\theta)\varepsilon&space;v-&space;v&space;^T&space;G(\phi)v" title="L(\theta) + L'(\theta)\varepsilon v- v ^T G(\phi)v" />

and set the derivative w.r.t. v, <img src="https://latex.codecogs.com/gif.latex?\frac{\varepsilon}{2v}&space;G^{-1}(\phi)L'(\theta)" title="\frac{\varepsilon}{2v} G^{-1}(\phi)L'(\theta)" />

<img src="https://latex.codecogs.com/gif.latex?G^{-1}(\phi)L'(\theta)" title="G^{-1}(\phi)L'(\theta)" /> is called natural gradient, which takes into accoutn the scale of different parameters, so it is parameteriation free.

## Relationship between Fisher Information metric and Hessian matrix of the log likelihood
The Fisher Information metric measures the geometrical property of the manifold the data lies in.
The Hessian matrix measures the local curvature at a point in the parameter space.
Therefore, the two should have certain connections.

The Fisher Information measures the expected information about a distribution by observing the  examples (also known as the score function, though i am not quite sure about this):
<!--E_x[\frac{\partial logP_\theta(x)}{\partial\theta}]-->
<img src="https://latex.codecogs.com/gif.latex?E_x[\frac{\partial&space;logP_\theta(x)}{\partial\theta}]" title="E_x[\frac{\partial logP_\theta(x)}{\partial\theta}]" />

In other words, the gradient w.r.t. each parameter accounts for how much each parameter contributes to the data.

The Fisher Information matrix is defined as:
<!--G(i,j)=E_x[\frac{\partial logP_\theta(x)}{\partial\theta_i}\frac{\partial logP_\theta(x)^T}{\partial\theta_j}]-->
<img src="https://latex.codecogs.com/gif.latex?G(i,j)=E_x[\frac{\partial&space;logP_\theta(x)}{\partial\theta_i}\frac{\partial&space;logP_\theta(x)^T}{\partial\theta_j}]" title="G(i,j)=E_x[\frac{\partial logP_\theta(x)}{\partial\theta_i}\frac{\partial logP_\theta(x)^T}{\partial\theta_j}]" />

[Fisher kernel](https://github.com/Scott-Alex/mixture_Fisher-Kernel) is the inner product of the Fisher score function weighted by the Fisher Information metric.
The parametrized model has been trained by maximum likelihood to approximate the data and represent the manifold of the data. So the Fisher kernel measures the distance of two data points on the manifold.

Now let us return to the Hessian matrix. The Hessian matrix is the second order derivative of the optimization function (in this case, the log likelihood function). By using chain rule, its formulation is generalized as:
<!--H_{i,j}(logP_\theta(x))=E_x[\frac{\partial logP_\theta(x)}{\partial \theta_i\partial \theta_j}]
=\partial \theta_iE_x[\frac{\partial logP_\theta(x)}{\partial \theta_j}]
=\partial \theta_iE_x[\frac{\partial_j P_\theta(x)}{P_\theta(x)}]\\
=E_x[\frac{\partial_i \partial_j P_\theta(x)P_\theta(x)-\partial_i P_\theta(x)\partial_j P_\theta(x)}{P^2_\theta(x)}]\\
=E_x[\frac{\partial_i \partial_j P_\theta(x)}{P_\theta(x)}-\frac{\partial_i P_\theta(x)\partial_j P_\theta(x)}{P^2_\theta(x)}]-->
<img src="https://latex.codecogs.com/gif.latex?H_{i,j}(logP_\theta(x))=E_x[\frac{\partial&space;logP_\theta(x)}{\partial&space;\theta_i\partial&space;\theta_j}]&space;=\partial&space;\theta_iE_x[\frac{\partial&space;logP_\theta(x)}{\partial&space;\theta_j}]&space;=\partial&space;\theta_iE_x[\frac{\partial_j&space;P_\theta(x)}{P_\theta(x)}]\\&space;=E_x[\frac{\partial_i&space;\partial_j&space;P_\theta(x)P_\theta(x)-\partial_i&space;P_\theta(x)\partial_j&space;P_\theta(x)}{P^2_\theta(x)}]\\&space;=E_x[\frac{\partial_i&space;\partial_j&space;P_\theta(x)}{P_\theta(x)}-\frac{\partial_i&space;P_\theta(x)\partial_j&space;P_\theta(x)}{P^2_\theta(x)}]" title="H_{i,j}(logP_\theta(x))=E_x[\frac{\partial logP_\theta(x)}{\partial \theta_i\partial \theta_j}] =\partial \theta_iE_x[\frac{\partial logP_\theta(x)}{\partial \theta_j}] =\partial \theta_iE_x[\frac{\partial_j P_\theta(x)}{P_\theta(x)}]\\ =E_x[\frac{\partial_i \partial_j P_\theta(x)P_\theta(x)-\partial_i P_\theta(x)\partial_j P_\theta(x)}{P^2_\theta(x)}]\\ =E_x[\frac{\partial_i \partial_j P_\theta(x)}{P_\theta(x)}-\frac{\partial_i P_\theta(x)\partial_j P_\theta(x)}{P^2_\theta(x)}]" />

Notice that
<!--E_x[\frac{\partial_i \partial_j P_\theta(x)}{P_\theta(x)}]\\
=\int_x \frac{\partial_i \partial_j P_\theta(x)}{P_\theta(x)} P_\theta(x)dx\\
=\int_x \partial_i \partial_j P_\theta(x)dx\\
=\partial_i \partial_j \int_x P_\theta(x)dx=0-->
<img src="https://latex.codecogs.com/gif.latex?E_x[\frac{\partial_i&space;\partial_j&space;P_\theta(x)}{P_\theta(x)}]\\&space;=\int_x&space;\frac{\partial_i&space;\partial_j&space;P_\theta(x)}{P_\theta(x)}&space;P_\theta(x)dx\\&space;=\int_x&space;\partial_i&space;\partial_j&space;P_\theta(x)dx\\&space;=\partial_i&space;\partial_j&space;\int_x&space;P_\theta(x)dx=0" title="E_x[\frac{\partial_i \partial_j P_\theta(x)}{P_\theta(x)}]\\ =\int_x \frac{\partial_i \partial_j P_\theta(x)}{P_\theta(x)} P_\theta(x)dx\\ =\int_x \partial_i \partial_j P_\theta(x)dx\\ =\partial_i \partial_j \int_x P_\theta(x)dx=0" />

Therefore, the Hessian matrix is reformulated as 
<!--H_{i,j}(logP_\theta(x))\\
=-E_x[\frac{\partial_i P_\theta(x)}{P_\theta(x)}\frac{\partial_j P_\theta(x)}{P_\theta(x)}]\\
=-G(i,j)-->
<img src="https://latex.codecogs.com/gif.latex?H_{i,j}(logP_\theta(x))\\&space;=-E_x[\frac{\partial_i&space;P_\theta(x)}{P_\theta(x)}\frac{\partial_j&space;P_\theta(x)}{P_\theta(x)}]\\&space;=-G(i,j)" title="H_{i,j}(logP_\theta(x))\\ =-E_x[\frac{\partial_i P_\theta(x)}{P_\theta(x)}\frac{\partial_j P_\theta(x)}{P_\theta(x)}]\\ =-G(i,j)" />

The conclusion is that the Hessian matrix of the log likelihood is equivalent to the negative Fisher Information metric.

The result is not surprising since the Fisher Information metric and the Hessian matrix both explains the curvature (or scales of different parameters) in the parameter space. (Note the parameter space concerns the loss function, while data space concerns the objective function.)


## Ref
https://hips.seas.harvard.edu/blog/2013/01/25/the-natural-gradient/

http://kvfrans.com/what-is-the-natural-gradient-and-where-does-it-appear-in-trust-region-policy-optimization/
