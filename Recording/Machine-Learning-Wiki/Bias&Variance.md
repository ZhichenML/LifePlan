# Bias Variance Trade-off 
**Keywords**: `bias variance decomposition`, `Uncertainty`, `biased Estimation of Variance`

## Bias Vanriance Decomposition
We review the bias and variance of a machine learning model :blush:. Assume we have a dataset coming from 
the generating mechanism 
<a href="https://www.codecogs.com/eqnedit.php?latex=f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f" title="f" /></a>. 
During the data is generated, we sample from it and get the data  
<a href="https://www.codecogs.com/eqnedit.php?latex=y=f&plus;\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y=f&plus;\epsilon" title="y=f+\epsilon" /></a>
, which, however, it corrupted by some inherent noise. We assume the noise is zero mean Gaussian:
<a href="https://www.codecogs.com/eqnedit.php?latex=\epsilon\sim&space;N(0,\sigma^2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\epsilon\sim&space;N(0,\sigma^2)" title="\epsilon\sim N(0,\sigma^2)" /></a>.
so, we have <a href="https://www.codecogs.com/eqnedit.php?latex=E[y]=f,&space;var[y]=\sigma^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E[y]=f,&space;var[y]=\sigma^2" title="E[y]=f, var[y]=\sigma^2" /></a>.

We formulate a model 
<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{f}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{f}" title="\hat{f}" /></a>
to approximate the data. 
In summary,

<a href="https://www.codecogs.com/eqnedit.php?latex=f\overset{\epsilon}{\rightarrow}y\rightarrow&space;\hat{f}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f\overset{\epsilon}{\rightarrow}y\rightarrow&space;\hat{f}" title="f\overset{\epsilon}{\rightarrow}y\rightarrow \hat{f}" /></a>

Since we have no knowledge about 
<a href="https://www.codecogs.com/eqnedit.php?latex=f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f" title="f" /></a>. 
Then, the approximation error is calculated by the discrepancy between the observed data and the model output:

<a href="https://www.codecogs.com/eqnedit.php?latex=E[(\hat{f}-y)^2]=E[\hat{f}^2]-2E[\hat{f}y]&plus;E[y^2]&space;\\&space;=&space;var[\hat{f}]&plus;E[\hat{f}]^2-2fE[\hat{f}]&space;&plus;&space;var[y]&space;&plus;&space;f^2\\&space;=var[\hat{f}]&plus;\sigma^2&plus;E[f-\hat{f}^2]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E[(\hat{f}-y)^2]=E[\hat{f}^2]-2E[\hat{f}y]&plus;E[y^2]&space;\\&space;=&space;var[\hat{f}]&plus;E[\hat{f}]^2-2fE[\hat{f}]&space;&plus;&space;var[y]&space;&plus;&space;f^2\\&space;=var[\hat{f}]&plus;\sigma^2&plus;E[f-\hat{f}^2]" title="E[(\hat{f}-y)^2]=E[\hat{f}^2]-2E[\hat{f}y]+E[y^2] \\ = var[\hat{f}]+E[\hat{f}]^2-2fE[\hat{f}] + var[y] + f^2\\ =var[\hat{f}]+\sigma^2+E[f-\hat{f}^2]" /></a>

The first term is the model uncertainty, which reflects out ignorance about the model parameters. 
It can be reduced as more data is available. The second term is the inherence noise, which is irreducible but could be estimated 
by the variance on the validation set [1]. The third term is the bias, which indicates the difference bettwen out model and the generating
mechanism. There is a fourth term, called model misspecification, that characterize the situation in which the test data distribution is different from that of 
the training data. In [1], the model misspecification is particially considered by Monte Carlo dropout in encoder-decoder system. 
It generates novel samples that is different from the training data by dropout contamination.

Despite the apparent mathematical meaning of bias and variance. Their useage is limited as far as I know, because it involves
repeated approximations of a model on a training set to get the average.

## Law of Total Variance

Just to remind the thought that the variance of a variable is the combination of two terms, i.e.,

<a href="https://www.codecogs.com/eqnedit.php?latex=var[x]=&space;E[var[x|y]]&space;&plus;&space;var[E[x|y]]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?var[x]=&space;E[var[x|y]]&space;&plus;&space;var[E[x|y]]" title="var[x]= E[var[x|y]] + var[E[x|y]]" /></a>

The first term is the average guess of the error of x over all **y**.
The second term is the variance of the **x** given all y.

For a new sample <a href="https://www.codecogs.com/eqnedit.php?latex=x^{\ast}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x^{\ast}" title="x^{\ast}" /></a>,

<a href="https://www.codecogs.com/eqnedit.php?latex=var[y^{\ast}|x^{\ast}]\\&space;=E[var[y^{\ast}|x^{\ast}]]&space;&plus;&space;var[E[y^{\ast}|x^{\ast}]]\\&space;=\sigma^2&space;&plus;&space;var[\hat{f}(x^{\ast})]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?var[y^{\ast}|x^{\ast}]\\&space;=E[var[y^{\ast}|x^{\ast}]]&space;&plus;&space;var[E[y^{\ast}|x^{\ast}]]\\&space;=\sigma^2&space;&plus;&space;var[\hat{f}(x^{\ast})]" title="var[y^{\ast}|x^{\ast}]\\ =E[var[y^{\ast}|x^{\ast}]] + var[E[y^{\ast}|x^{\ast}]]\\ =\sigma^2 + var[\hat{f}(x^{\ast})]" /></a>

which is the inherent noise and the model uncertainty.


## Expectation of Maximum Likelihood Estimation
Suppose we approximate the dataset having <a href="https://www.codecogs.com/eqnedit.php?latex=N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N" title="N" /></a> 
samples (such as by maximum likelihood estimation) and obtained a set of repetitions.
Let the mean and variance of the model output be 

<a href="https://www.codecogs.com/eqnedit.php?latex=\mu_{ML}=\frac{1}{N}\sum^{N}_{i=1}y^{i},&space;\sigma^2{}_{ML}=\frac{1}{N}\sum_{i=1}^{N}(y^{i}-\mu_{ML})^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu_{ML}=\frac{1}{N}\sum^{N}_{i=1}y^{i},&space;\sigma^2{}_{ML}=\frac{1}{N}\sum_{i=1}^{N}(y^{i}-\mu_{ML})^2" title="\mu_{ML}=\frac{1}{N}\sum^{N}_{i=1}y^{i}, \sigma^2{}_{ML}=\frac{1}{N}\sum_{i=1}^{N}(y^{i}-\mu_{ML})^2" /></a>

Then, 
<!--E[\mu_{ML}]=E[\frac{1}{N}\sum^{N}_{i=1}y^{i}]=\mu, \\
var[\mu_{ML}]=var[\frac{1}{N}\sum^{N}_{i=1}y^{i}]=\frac{1}{N^2}var[\sum^{N}_{i=1}y^{i}]=\frac{1}{N^2}*N*\sigma^2=\frac{1}{N}\sigma^2
-->
<a href="https://www.codecogs.com/eqnedit.php?latex=E[\mu_{ML}]=E[\frac{1}{N}\sum^{N}_{i=1}y^{i}]=\mu,&space;\\&space;var[\mu_{ML}]=var[\frac{1}{N}\sum^{N}_{i=1}y^{i}]=\frac{1}{N^2}var[\sum^{N}_{i=1}y^{i}]=\frac{1}{N^2}*N*\sigma^2=\frac{1}{N}\sigma^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E[\mu_{ML}]=E[\frac{1}{N}\sum^{N}_{i=1}y^{i}]=\mu,&space;\\&space;var[\mu_{ML}]=var[\frac{1}{N}\sum^{N}_{i=1}y^{i}]=\frac{1}{N^2}var[\sum^{N}_{i=1}y^{i}]=\frac{1}{N^2}*N*\sigma^2=\frac{1}{N}\sigma^2" title="E[\mu_{ML}]=E[\frac{1}{N}\sum^{N}_{i=1}y^{i}]=\mu, \\ var[\mu_{ML}]=var[\frac{1}{N}\sum^{N}_{i=1}y^{i}]=\frac{1}{N^2}var[\sum^{N}_{i=1}y^{i}]=\frac{1}{N^2}*N*\sigma^2=\frac{1}{N}\sigma^2" /></a>

The expectation of the variance is 
<!--E[\sigma^2_{ML}]=E[\frac{1}{N}\sum^{N}_{i=1}(y^{i}-\mu_{ML})^2] \\
=\frac{1}{N}E[\sum^{N}_{i=1}(y^{i}-\mu_{ML})^2]\\
=\frac{1}{N}E[\sum^{N}_{i=1}(y^{i})^2-2\sum^{N}_{i=1}\mu_{ML}*y^i + \sum^{N}_{i=1}\mu_{ML}^2] \\
=\frac{1}{N}E[\sum^{N}_{i=1}(y^{i})^2-2\sum^{N}_{i=1}\mu_{ML}^2 + \sum^{N}_{i=1}\mu_{ML}^2] \\
=\frac{1}{N}E[\sum^{N}_{i=1}(y^{i})^2]-E[\mu_{ML}^2] 
=\frac{1}{N} E[\sum^{N}_{i=1}(y^{i})^2]-(var[\mu_{ML}]+E[\mu_{ML}]^2) \\
= E[y^2]-\frac{1}{N}\sigma^2- \mu^2\\
= var[y]+E[Y]^2-\frac{1}{N}\sigma^2-\mu^2\\
= \sigma^2+\mu^2-\frac{1}{N}\sigma^2-\mu^2\\
=\frac{N-1}{N}\sigma^2-->

<a href="https://www.codecogs.com/eqnedit.php?latex=E[\sigma^2_{ML}]=E[\frac{1}{N}\sum^{N}_{i=1}(y^{i}-\mu_{ML})^2]&space;\\&space;=\frac{1}{N}E[\sum^{N}_{i=1}(y^{i}-\mu_{ML})^2]\\&space;=\frac{1}{N}E[\sum^{N}_{i=1}(y^{i})^2-2\sum^{N}_{i=1}\mu_{ML}*y^i&space;&plus;&space;\sum^{N}_{i=1}\mu_{ML}^2]&space;\\&space;=\frac{1}{N}E[\sum^{N}_{i=1}(y^{i})^2-2\sum^{N}_{i=1}\mu_{ML}^2&space;&plus;&space;\sum^{N}_{i=1}\mu_{ML}^2]&space;\\&space;=\frac{1}{N}E[\sum^{N}_{i=1}(y^{i})^2]-E[\mu_{ML}^2]&space;\\&space;=\frac{1}{N}&space;E[\sum^{N}_{i=1}(y^{i})^2]-(var[\mu_{ML}]&plus;E[\mu_{ML}]^2)&space;\\&space;=&space;E[y^2]-\frac{1}{N}\sigma^2-&space;\mu^2\\&space;=&space;var[y]&plus;E[y]^2-\frac{1}{N}\sigma^2-\mu^2\\&space;=&space;\sigma^2&plus;\mu^2-\frac{1}{N}\sigma^2-\mu^2\\&space;=\frac{N-1}{N}\sigma^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E[\sigma^2_{ML}]=E[\frac{1}{N}\sum^{N}_{i=1}(y^{i}-\mu_{ML})^2]&space;\\&space;=\frac{1}{N}E[\sum^{N}_{i=1}(y^{i}-\mu_{ML})^2]\\&space;=\frac{1}{N}E[\sum^{N}_{i=1}(y^{i})^2-2\sum^{N}_{i=1}\mu_{ML}*y^i&space;&plus;&space;\sum^{N}_{i=1}\mu_{ML}^2]&space;\\&space;=\frac{1}{N}E[\sum^{N}_{i=1}(y^{i})^2-2\sum^{N}_{i=1}\mu_{ML}^2&space;&plus;&space;\sum^{N}_{i=1}\mu_{ML}^2]&space;\\&space;=\frac{1}{N}E[\sum^{N}_{i=1}(y^{i})^2]-E[\mu_{ML}^2]&space;\\&space;=\frac{1}{N}&space;E[\sum^{N}_{i=1}(y^{i})^2]-(var[\mu_{ML}]&plus;E[\mu_{ML}]^2)&space;\\&space;=&space;E[y^2]-\frac{1}{N}\sigma^2-&space;\mu^2\\&space;=&space;var[y]&plus;E[y]^2-\frac{1}{N}\sigma^2-\mu^2\\&space;=&space;\sigma^2&plus;\mu^2-\frac{1}{N}\sigma^2-\mu^2\\&space;=\frac{N-1}{N}\sigma^2" title="E[\sigma^2_{ML}]=E[\frac{1}{N}\sum^{N}_{i=1}(y^{i}-\mu_{ML})^2] \\ =\frac{1}{N}E[\sum^{N}_{i=1}(y^{i}-\mu_{ML})^2]\\ =\frac{1}{N}E[\sum^{N}_{i=1}(y^{i})^2-2\sum^{N}_{i=1}\mu_{ML}*y^i + \sum^{N}_{i=1}\mu_{ML}^2] \\ =\frac{1}{N}E[\sum^{N}_{i=1}(y^{i})^2-2\sum^{N}_{i=1}\mu_{ML}^2 + \sum^{N}_{i=1}\mu_{ML}^2] \\ =\frac{1}{N}E[\sum^{N}_{i=1}(y^{i})^2]-E[\mu_{ML}^2] \\ =\frac{1}{N} E[\sum^{N}_{i=1}(y^{i})^2]-(var[\mu_{ML}]+E[\mu_{ML}]^2) \\ = E[y^2]-\frac{1}{N}\sigma^2- \mu^2\\ = var[y]+E[y]^2-\frac{1}{N}\sigma^2-\mu^2\\ = \sigma^2+\mu^2-\frac{1}{N}\sigma^2-\mu^2\\ =\frac{N-1}{N}\sigma^2" /></a>

That is, after enough repetition of learning on a dataset, the mean is unbiased whereas the variance is biased by <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{1}{N}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{1}{N}" title="\frac{1}{N}" /></a>.
However, if the training set size is large enough, this bias does not seem to make a big difference.

## Reference
1. [Zhu, Lingxue, and Nikolay Laptev. "Deep and Confident Prediction for Time Series at Uber." Data Mining Workshops (ICDMW), 2017 IEEE International Conference on. IEEE, 
2017.](https://arxiv.org/abs/1709.01907)
2. Pattern Recognition and Machine Learning, Cha 1 (Equation 1.58).
