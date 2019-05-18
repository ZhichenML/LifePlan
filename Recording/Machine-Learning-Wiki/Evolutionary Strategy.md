
# Evolution strategy- heuristic optimizing method as a special SGD 

ES has four main advantages:
* Gradient free optimization. It helps avoid gradient vanishing or explosion problems in optimizing neural networks.
* Hyperparameter rubostness. The algorithm has some parameters such as population size, cross over/ mutation rate, number of iteration, which are not sensitive to the performance.
* Easy to optimize hypeparameters of NN, e.g. network size.
* Naturally distributed. It involves little interaction among separated workers.

## Approximating the Gradient 
Evolutionary strategy is actually an approximation of the gradient of a function f, which can be shown be 2-order taylor expansion:

<img src="https://latex.codecogs.com/gif.latex?f(x&plus;\theta)=f(x)&plus;\theta&space;f'(x)&plus;\frac{\theta^2}{2}f''(x)&plus;O(\theta^3)" title="f(x+\theta)=f(x)+\theta f'(x)+\frac{\theta^2}{2}f''(x)+O(\theta^3)" />

Multiply <img src="https://latex.codecogs.com/gif.latex?\theta" title="\theta" /> and take expectation, we obtain

<img src="https://latex.codecogs.com/gif.latex?E_{\theta\sim&space;N}[\theta&space;f(x&plus;\theta)]=E_{\theta\sim&space;N}[\theta&space;f(x)&plus;\theta^2&space;f'(x)&plus;\frac{\theta^3}{2}f''(x)&plus;O(\theta^4)]&space;\\&space;\approx&space;E_{\theta\sim&space;N}[\theta^2&space;f'(x)]&space;=&space;\sigma^2&space;f'(x)" title="E_{\theta\sim N}[\theta f(x+\theta)]=E_{\theta\sim N}[\theta f(x)+\theta^2 f'(x)+\frac{\theta^3}{2}f''(x)+O(\theta^4)] \\ \approx E_{\theta\sim N}[\theta^2 f'(x)] = \sigma^2 f'(x)" />

Therefore,

<img src="https://latex.codecogs.com/gif.latex?f'(x)=\frac{1}{\sigma^2}E_{\theta&space;\sim&space;N}[\theta&space;f(x&plus;\theta)]&space;=&space;\frac{1}{S&space;\sigma^2}\sum_{n}\theta^{(s)}&space;f(x&plus;\theta^{(s))})" title="f'(x)=\frac{1}{\sigma^2}E_{\theta \sim N}[\theta f(x+\theta)] = \frac{1}{S \sigma^2}\sum_{n}\theta^{(s)} f(x+\theta^{(s))})" />

So Evolutionary strategy perturbs the current parameters by additive Gaussian noise, and evalues the function values of the contaminated parameters.
Finally, the function values are combined to form a gradient estimate and take a step towards the opposite direction.

## Low Cost Parallel
Distributed SGD often communicates with workers or central parameter servers to update the parameters. However, when the parameter space is 
huge, this may become a time consuming step and slow down everything.

ES calculates the function values <img src="https://latex.codecogs.com/gif.latex?f(x&plus;\theta)" title="f(x+\theta)" /> to approximate the
gradient. Each worker calculates its own function values and then communicate only the single scalar, which could be extremely fast. Since 
all the workers can have the random noise generating mechanism in local file, there is no need to communicate the <img src="https://latex.codecogs.com/gif.latex?\theta" title="\theta" />.

## Non-differential Function
For SGD, the gradient is justified as:

<img src="https://latex.codecogs.com/gif.latex?f'_{w}(x;w)=\frac{\partial}{\partial&space;w}&space;E_{x}[f(x;w)]=E_{x}[\frac{\partial}{\partial&space;w}f(x;w))]" title="f'_{w}(x;w)=\frac{\partial}{\partial w} E_{x}[f(x;w)]=E_{x}[\frac{\partial}{\partial w}f(x;w))]" />

However, many practical optimizing functions are not differentiable, examples inlcude:
* The policy function of Reinforcement learning
* The structure of Neural networks
* variational autoencoder with discrete latent variables (Jang et al, 2016)

ES avoids this problem because it only involves funcation evaluation. You can take the expectation with respect to x and 
<img src="https://latex.codecogs.com/gif.latex?\theta" title="\theta" />, instead of the differentiation:
<img src="https://latex.codecogs.com/gif.latex?E_{x}[&space;E_{\varepsilon&space;\sim&space;N}[\varepsilon&space;f(\theta&plus;\varepsilon;x)]]" title="E_{x}[ E_{\varepsilon \sim N}[\varepsilon f(\theta+\varepsilon;x)]]" />

## Second-order Gradient
ES can also be extended to approximate second order gradients:
<img src="https://latex.codecogs.com/gif.latex?E_{\varepsilon&space;\sim&space;N}[(\frac{\varepsilon&space;^2}{\sigma&space;^2}-1)\varepsilon&space;f(\theta&plus;\varepsilon&space;)]&space;\\&space;=\frac{1}{\sigma^2}E_{\varepsilon&space;\sim&space;N}[\varepsilon^2f(x)&plus;\varepsilon^3f'(x)&plus;\frac{\varepsilon^4}{2}f''(x&plus;\varepsilon)]&space;-E_{\varepsilon&space;\sim&space;N}[f(x)&plus;\varepsilon&space;f'(x)&plus;\frac{\varepsilon^2}{2}&space;f''(x))]\\&space;=\frac{1}{\sigma^2}(\sigma^2&space;f(x)&space;&plus;&space;\frac{3\sigma^4}{2}f''(x))-(f(x)&plus;\frac{\sigma^2}{2}f''(x))\\&space;=\sigma^2&space;f''(x)" title="E_{\varepsilon \sim N}[(\frac{\varepsilon ^2}{\sigma ^2}-1)\varepsilon f(\theta+\varepsilon )] \\ =\frac{1}{\sigma^2}E_{\varepsilon \sim N}[\varepsilon^2f(x)+\varepsilon^3f'(x)+\frac{\varepsilon^4}{2}f''(x+\varepsilon)] -E_{\varepsilon \sim N}[f(x)+\varepsilon f'(x)+\frac{\varepsilon^2}{2} f''(x))]\\ =\frac{1}{\sigma^2}(\sigma^2 f(x) + \frac{3\sigma^4}{2}f''(x))-(f(x)+\frac{\sigma^2}{2}f''(x))\\ =\sigma^2 f''(x)" />

This estimate may be biased due to the taylor expansion and has a high vairance due to sampling. Once you have calculated the fucntion
values, the second order gradient can be employed for free.

## Reference
* >Salimans T, Ho J, Chen X, et al. Evolution strategies as a scalable alternative to reinforcement learning[J]. arXiv preprint arXiv:1703.03864, 2017.
* >https://blog.openai.com/evolution-strategies/
