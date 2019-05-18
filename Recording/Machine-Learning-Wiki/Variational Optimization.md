In this note, we will first introduce the variational optimizaiton, then we will infer the conclusion that Evolution strategy is 
a Gaussian perturbed variational optimization.

Variation Optimization is based on the principle:

<img src="https://latex.codecogs.com/gif.latex?min_{x}f(x)\leq&space;E_{x}[f(x)]_{P(x|\theta)}" title="min_{x}f(x)\leq E_{x}[f(x)]_{P(x|\theta)}" />

That is, the minimum of a function is lower than the average of the funciton.
Based on this observation, we can instead minimize the unpper bound for non-differential function or discrete x. Let 

<img src="https://latex.codecogs.com/gif.latex?U(\theta)&space;=&space;E_{x}[f(x)]_{P(x|\theta)}" title="U(\theta) = E_{x}[f(x)]_{P(x|\theta)}" />,

then our task is:

<img src="https://latex.codecogs.com/gif.latex?min_{\theta}U(\theta)" title="min_{\theta}U(\theta)" />

Calculate the gradient:

<img src="https://latex.codecogs.com/gif.latex?\frac{\partial}{\partial&space;\theta}U(\theta)&space;=&space;\frac{\partial}{\partial&space;\theta}&space;\int&space;f(x)P(x|\theta)&space;dx&space;=\int&space;f(x)\frac{\partial}{\partial&space;\theta}P(x|\theta)\frac{P(x|\theta)}{P(x|\theta)}&space;dx\\&space;=\int&space;f(x)\frac{\partial}{\partial&space;\theta}&space;logP(x|\theta)&space;P(x|\theta)dx=E_{x}[f(x)\frac{\partial}{\partial&space;\theta}logP(x|\theta)]_{P(x|\theta)}" title="\frac{\partial}{\partial \theta}U(\theta) = \frac{\partial}{\partial \theta} \int f(x)P(x|\theta) dx =\int f(x)\frac{\partial}{\partial \theta}P(x|\theta)\frac{P(x|\theta)}{P(x|\theta)} dx\\ =\int f(x)\frac{\partial}{\partial \theta} logP(x|\theta) P(x|\theta)dx=E_{x}[f(x)\frac{\partial}{\partial \theta}logP(x|\theta)]_{P(x|\theta)}" />
<!--\frac{\partial}{\partial \theta}U(\theta) = \frac{\partial}{\partial \theta} \int f(x)P(x|\theta) dx
=\int f(x)\frac{\partial}{\partial \theta}P(x|\theta)\frac{P(x|\theta)}{P(x|\theta)} dx\\
=\int f(x)\frac{\partial}{\partial \theta} logP(x|\theta) P(x|\theta)dx=E_{x}[f(x)\frac{\partial}{\partial \theta}logP(x|\theta)]_{P(x|\theta)}-->

Let <img src="https://latex.codecogs.com/gif.latex?P(x|\theta)" title="P(x|\theta)" /> be zero-mean Gaussian, then (assuming single variable but can be generalized to vectors)
<!--U(u)=\frac{1}{\sqrt{2 \pi \sigma^2}} \int exp^{\frac{-(x-u)^2}{2\sigma^2}}f(x)dx-->

<img src="https://latex.codecogs.com/gif.latex?U(u)=\frac{1}{\sqrt{2&space;\pi&space;\sigma^2}}&space;\int&space;exp^{\frac{-(x-u)^2}{2\sigma^2}}f(x)dx" title="U(u)=\frac{1}{\sqrt{2 \pi \sigma^2}} \int exp^{\frac{-(x-u)^2}{2\sigma^2}}f(x)dx" />

and the gradient is
<!--\frac{\partial}{\partial u} U(u)=\frac{1}{\sqrt{2 \pi \sigma^2}} \int \frac{\partial}{\partial u}exp^{\frac{-(x-u)^2}{2\sigma^2}}f(x)dx \\
=\frac{1}{\sigma^2}E_{x}[(x-u)f(x)]_{x\sim N(u,\sigma^2)}-->
<img src="https://latex.codecogs.com/gif.latex?\frac{\partial}{\partial&space;u}&space;U(u)=\frac{1}{\sqrt{2&space;\pi&space;\sigma^2}}&space;\int&space;\frac{\partial}{\partial&space;u}exp^{\frac{-(x-u)^2}{2\sigma^2}}f(x)dx&space;\\&space;=\frac{1}{\sigma^2}E_{x}[(x-u)f(x)]_{x\sim&space;N(u,\sigma^2)}" title="\frac{\partial}{\partial u} U(u)=\frac{1}{\sqrt{2 \pi \sigma^2}} \int \frac{\partial}{\partial u}exp^{\frac{-(x-u)^2}{2\sigma^2}}f(x)dx \\ =\frac{1}{\sigma^2}E_{x}[(x-u)f(x)]_{x\sim N(u,\sigma^2)}" />

The above result is equvelent to
<!--\frac{1}{\sigma^2} E_{\varepsilon}[\varepsilon f(u+\varepsilon)]_{\varepsilon\sim N(0,\sigma^2)}-->
<img src="https://latex.codecogs.com/gif.latex?\frac{1}{\sigma^2}&space;E_{\varepsilon}[\varepsilon&space;f(u&plus;\varepsilon)]_{\varepsilon\sim&space;N(0,\sigma^2)}" title="\frac{1}{\sigma^2} E_{\varepsilon}[\varepsilon f(u+\varepsilon)]_{\varepsilon\sim N(0,\sigma^2)}" />

It resembles the mechanism of [Evolutionary Strategy](https://github.com/Scott-Alex/Machine-Learning-Wiki/blob/master/Evolutionary%20Strategy.md)

From the above result, we find that the Evolutionary strategy is connected to variational optimization with <img src="https://latex.codecogs.com/gif.latex?P(x|\theta)" title="P(x|\theta)" /> as Guassian.
Also note that variational optimization optimises <img src="https://latex.codecogs.com/gif.latex?\theta" title="\theta" /> rather that <img src="https://latex.codecogs.com/gif.latex?x" title="x" />.
The main advantage of variational optimization over Evolutionary strategy is that VO is more flexible to different distributions 
and provides a principled approach to optimize the variance <img src="https://latex.codecogs.com/gif.latex?\sigma^2" title="\sigma^2" />.

Reference
* > Variational Optimization, Joe Staines, David Barber, arXiv:1212.4507.
