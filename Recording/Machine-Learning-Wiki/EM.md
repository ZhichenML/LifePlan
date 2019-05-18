# EM algorithm

**EM algorithm** is an useful to maximize the likelihood of data where unobserved hidden variables exist, such as the hidden label for different distributions generating the data. Due to the hidden variables, the likelihood function involves log(sum) form, which makes the gradient less splendid.

![](https://github.com/Scott-Alex/Learning-material/blob/master/fig/EMequ.png)
<!--L(\theta) = logP(y|\theta)=log\sum_{z}P(z|\theta)P(y|z,\theta) \\
=log\sum_{z}Q(z)\frac{P(z|\theta)P(y|z,\theta)}{Q(z))} \\
\geq \sum_{z}Q(z)log\frac{P(z|\theta)P(y|z,\theta)}{Q(z))} \\
where\ the\ equation\ holds\ iff\  \frac{P(z|\theta)P(y|z,\theta)}{Q(z))}=c,\\
and\ \sum_{z}Q(z) = 1, so\\
Q(z)=\frac{P(z|\theta)P(y|z,\theta)}{\sum_{z}P(z|\theta)P(y|z,\theta)}=P(z|y,\theta)-->


We find a lower bound by using Jessen equation, and find the equation holds when the last equation holds.
Then the EM algorithm goes like this:
* E: calculate the posterior probability of hidden variable, because this makes the lower bound equals the previous likelihood.
* M: maximize the likelihood (the lower bound) by optimizing \theta.

For example, in K-means clustering, the hidden vairable is the cluster-label for each instance, the model parameter is the cluster center.

To summarize, we first adjust the hidden variable to make the lower bound as strong as the previous likelihood, then adjust the model parameter to maximize the current lower bound.

## Conclusion
The EM algorithm is usually useful to learn discrete latent variable models, with a toletable numble of hidden variables.
The optimization objective is 
<!--L = E_z[log\frac{p_\theta(x,z)}{q(z|x,\theta)}]_{q(z|x,\theta)}-->
<img src="https://latex.codecogs.com/gif.latex?L&space;=&space;E_z[log\frac{p_\theta(x,z)}{q(z|x,\theta)}]_{q(z|x,\theta)}" title="L = E_z[log\frac{p_\theta(x,z)}{q(z|x,\theta)}]_{q(z|x,\theta)}" />

The meaning is to maximize the likelihood of observable and hidden variables, as well as maximizing the entropy of hiiden variables, to 
make the latent distribution be spread.

If the <img src="https://latex.codecogs.com/gif.latex?q(z|x,\theta)" title="q(z|x,\theta)" /> is not easy to calculate, sampling algorithm
is usually needed, leading to variational inference.
