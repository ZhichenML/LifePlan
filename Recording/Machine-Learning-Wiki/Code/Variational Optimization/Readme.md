Variational optimization is to sample around the current solution subject to certain distributions. 
The parameters of the assumed distribution is optimized by gradient descent.

There are several parameters: learning rate, sample size, initial values (mean and std for Gaussian), also the range of the function 
value influences the update strength.
> Parameter suggestion: set large sd, sample size, adjust the learning rate.

Assume the sample distribution is Gaussian distribution:
<!--P(x|\mu,\sigma^2) = \frac{1}{\sqrt{2*\pi}\sigma}exp\{-\frac{(x-\mu)^2}{2*\sigma^2}\}-->
<img src="https://latex.codecogs.com/gif.latex?P(x|\mu,\sigma^2)&space;=&space;\frac{1}{\sqrt{2*\pi}\sigma}exp\{-\frac{(x-\mu)^2}{2*\sigma^2}\}" title="P(x|\mu,\sigma^2) = \frac{1}{\sqrt{2*\pi}\sigma}exp\{-\frac{(x-\mu)^2}{2*\sigma^2}\}" />


The approximation gradicent is:
<!--f'(x) = \frac{1}{\sigma^2}E[f(x)log(P(x|\mu,\sigma^2)]_{P(x|\mu,\sigma^2}=\frac{1}{S\sigma^2}\sum_{s}[f(x^{(s)})log(P(x^{(s)}|\mu,\sigma^2)]-->
<img src="https://latex.codecogs.com/gif.latex?f'(x)&space;=&space;\frac{1}{\sigma^2}E[f(x)log(P(x|\mu,\sigma^2)]_{P(x|\mu,\sigma^2}=\frac{1}{S\sigma^2}\sum_{s}[f(x^{(s)})log(P(x^{(s)}|\mu,\sigma^2)]" title="f'(x) = \frac{1}{\sigma^2}E[f(x)log(P(x|\mu,\sigma^2)]_{P(x|\mu,\sigma^2}=\frac{1}{S\sigma^2}\sum_{s}[f(x^{(s)})log(P(x^{(s)}|\mu,\sigma^2)]" />

where the log term is
<!--logP(x|\mu,\sigma^2) = -log(\sqrt{(2\pi)})-log\sigma- \frac{(x-\mu)^2}{2\sigma^2}-->
<img src="https://latex.codecogs.com/gif.latex?logP(x|\mu,\sigma^2)&space;=&space;-log(\sqrt{(2\pi)})-log\sigma-&space;\frac{(x-\mu)^2}{2\sigma^2}" title="logP(x|\mu,\sigma^2) = -log(\sqrt{(2\pi)})-log\sigma- \frac{(x-\mu)^2}{2\sigma^2}" />

Then,
<!--\frac{\partial logP(x|\mu,\sigma^2)}{\partial\mu} = \frac{(x-\mu)}{\sigma^2}-->
<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;logP(x|\mu,\sigma^2)}{\partial\mu}&space;=&space;\frac{(x-\mu)}{\sigma^2}" title="\frac{\partial logP(x|\mu,\sigma^2)}{\partial\mu} = \frac{(x-\mu)}{\sigma^2}" />
<!--\frac{\partial logP(x|\mu,\sigma^2)}{\partial\sigma} = -\frac{1}{\sigma}+\frac{(x-\mu)^2}{\sigma^3}-->
<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;logP(x|\mu,\sigma^2)}{\partial\sigma}&space;=&space;-\frac{1}{\sigma}&plus;\frac{(x-\mu)^2}{\sigma^3}" title="\frac{\partial logP(x|\mu,\sigma^2)}{\partial\sigma} = -\frac{1}{\sigma}+\frac{(x-\mu)^2}{\sigma^3}" />

If the <img src="https://latex.codecogs.com/gif.latex?\sigma" title="\sigma" /> is too small (maybe after a few iterations), the gradient 
would be unstable. Therefore, let <img src="https://latex.codecogs.com/gif.latex?\beta&space;=&space;2log\sigma" title="\beta = 2log\sigma" />, then
<!--\frac{\partial logP(x|\mu,\sigma^2)}{\partial\beta} = -\frac{1}{2}-exp\{-\beta\}\frac{(x-\mu)^2}{2}=-\frac{1}{2}(1+exp\{-\beta\}(x-\mu)^2)-->
<img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;logP(x|\mu,\sigma^2)}{\partial\beta}&space;=&space;-\frac{1}{2}-exp\{-\beta\}\frac{(x-\mu)^2}{2}=-\frac{1}{2}(1&plus;exp\{-\beta\}(x-\mu)^2)" title="\frac{\partial logP(x|\mu,\sigma^2)}{\partial\beta} = -\frac{1}{2}-exp\{-\beta\}\frac{(x-\mu)^2}{2}=-\frac{1}{2}(1+exp\{-\beta\}(x-\mu)^2)" />

Now, we can set an initial value, then sample points near the solution with Gaussian distribution, evaluate the function values of samples, calculate the approximated gradients, update the solution, until the prefixe maximum iteration is met.

The figure below is the result for a linear regression problem, wehre the yellow is the SGD updates and the red is the VO updates.

![](https://github.com/Scott-Alex/Machine-Learning-Wiki/blob/master/Code/Variational%20Optimization/vo.png)

If you set the sample size as 100 or even larger, you will find the VO updates is very close to the SGD updates.

The standard deviation shrinks quickly.

![](https://github.com/Scott-Alex/Machine-Learning-Wiki/blob/master/Code/Variational%20Optimization/std.png)

Here we show the result of a non-differential function.

![](https://github.com/Scott-Alex/Machine-Learning-Wiki/blob/master/Code/Variational%20Optimization/non_diff.png)

Below is the result of a one dimensional example of a non-differential with discrete variable.

![](https://github.com/Scott-Alex/Machine-Learning-Wiki/blob/master/Code/Variational%20Optimization/dicrete.png)

The parameter tuning for this problem is a little tricky. In summary, you have to find a proper sd to cover the potential search range,
also adjust the learning rate to prevent tiny updates. I found the algorithm may not converge is the parameter is not properly set.

> If you do not optimize sd, then it is Evolutionary strategy.

Here is a comparison between differetiable and non-differentiable function.

![](https://github.com/Scott-Alex/Machine-Learning-Wiki/blob/master/Code/Variational%20Optimization/diffentiableFunc.png)
![](https://github.com/Scott-Alex/Machine-Learning-Wiki/blob/master/Code/Variational%20Optimization/diffentiableFunc_sample.png)

Non-differentiable function.

![](https://github.com/Scott-Alex/Machine-Learning-Wiki/blob/master/Code/Variational%20Optimization/nondiffentiableFunc.png)
![](https://github.com/Scott-Alex/Machine-Learning-Wiki/blob/master/Code/Variational%20Optimization/nondiffentiableFunc_sample.png)

It is observed that despite the origianl function is non-differentiable w.r.t x, it is most of the time differentiable to the parameters
of the distribution. However, if the function is constant for a large range, the sampling may not contribute to the optimization much,
as the approximation gradient unbiased to 0. In this case, large sd is needed to jump out the constant range.

The standard deviation controls the bias/variance trade-off. As the optimization goes, the sd will become smaller, increasing the 
variance and may not jump out of the local optima. Worsely, the improper sd may prevent us from finding the narrow global optima.
The following figure domonstrates its influence.

![](https://github.com/Scott-Alex/Machine-Learning-Wiki/blob/master/Code/Variational%20Optimization/sigma_influence_function.png)

![](https://github.com/Scott-Alex/Machine-Learning-Wiki/blob/master/Code/Variational%20Optimization/sigma_influence_sampling.png)

![](https://github.com/Scott-Alex/Machine-Learning-Wiki/blob/master/Code/Variational%20Optimization/sigma_influence_sigma.png)

