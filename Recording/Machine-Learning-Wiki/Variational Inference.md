# Variational Inference

Previously, we have gone through the rational of variational optimization and its implementation. 
Variational Inference is useful in approximate the posteriors in latent variable models.
I am not fully familiar with this field, thus i might gradually incorporate more information about this topic in this post.

Latent variable model has been actively investigated for its effectiveness of modeling the latent structure from data.
One of the key problem is the inference of the posterior probability of latent variables, which is usually intractable and resorts to 
approximation methods. One of which is variational inference. The idea is to find a member from a family of simple distributions.
Formally, we have

<!--KL(q(z|\lambda)||p(z|x)) = E_q[log\frac{q(z|\lambda)}{p(z|x)}]\\
=E_q[logq(z|\lambda)-logp(x,z)] + logp(x) \geq 0-->
<img src="https://latex.codecogs.com/gif.latex?KL(q(z|\lambda)||p(z|x))&space;=&space;E_q[log\frac{q(z|\lambda)}{p(z|x)}]\\&space;=E_q[logq(z|\lambda)-logp(x,z)]&space;&plus;&space;logp(x)&space;\geq&space;0" title="KL(q(z|\lambda)||p(z|x)) = E_q[log\frac{q(z|\lambda)}{p(z|x)}]\\ =E_q[logq(z|\lambda)-logp(x,z)] + logp(x) \geq 0" />

So,
<!--logp(x) \geq E_q[logp(x,z)-logq(z|\lambda)]-->
<img src="https://latex.codecogs.com/gif.latex?logp(x)&space;\geq&space;E_q[logp(x,z)-logq(z|\lambda)]" title="logp(x) \geq E_q[logp(x,z)-logq(z|\lambda)]" />

Therefore, we find the Evidence lower bound (ELBO). The origianl objecive is to minimize the difference between <img src="https://latex.codecogs.com/gif.latex?p(z|x)" title="p(z|x)" />
and <img src="https://latex.codecogs.com/gif.latex?q(z|\lambda)" title="q(z|\lambda)" />. After transformation, we get an equvalent objective,
that is to maximize the ELBO, which will minimize the difference. 

We can also find the bound bu maximizing likelihood:
<!--logp(x|\lambda) = log \int p(x,z|\lambda)dz=log \int q(z|x)\frac{p(x,z|\lambda)}{q(z|x)}dz\\
\geq \int q(z|x)log\frac{p(x,z|\lambda)}{q(z|x)}dz = E_z[log\frac{p(x,z|\lambda)}{q(z|x)}]_{q(z|x)}-->
<img src="https://latex.codecogs.com/gif.latex?logp(x|\lambda)&space;=&space;log&space;\int&space;p(x,z|\lambda)dz=log&space;\int&space;q(z|x)\frac{p(x,z|\lambda)}{q(z|x)}dz\\&space;\geq&space;\int&space;q(z|x)log\frac{p(x,z|\lambda)}{q(z|x)}dz&space;=&space;E_z[log\frac{p(x,z|\lambda)}{q(z|x)}]_{q(z|x)}" title="logp(x|\lambda) = log \int p(x,z|\lambda)dz=log \int q(z|x)\frac{p(x,z|\lambda)}{q(z|x)}dz\\ \geq \int q(z|x)log\frac{p(x,z|\lambda)}{q(z|x)}dz = E_z[log\frac{p(x,z|\lambda)}{q(z|x)}]_{q(z|x)}" />

This form is exactly the bound of EM algorithm (which is applicable to a low number of discrete  hidden vaiables).

The bound can also be reconstructed as:
<!--\int q(z|x)log\frac{p_\lambda(x|z)p(z)}{q(z|x)}dz = \int q(z|x)logp_\lambda(x|z) - KL(q(z|x)||p(z))-->
<img src="https://latex.codecogs.com/gif.latex?\int&space;q(z|x)log\frac{p_\lambda(x|z)p(z)}{q(z|x)}dz&space;=&space;\int&space;q(z|x)logp_\lambda(x|z)&space;-&space;KL(q(z|x)||p(z))" title="\int q(z|x)log\frac{p_\lambda(x|z)p(z)}{q(z|x)}dz = \int q(z|x)logp_\lambda(x|z) - KL(q(z|x)||p(z))" />

The former term is the reconstructed error of data, used is variational auto-encoder. 
The second term is the KL distance between the hidden variable distribution and the unit Gaussian distribution, with the goal
of control the variance to provide more informaiton, similar to diversity regularization.

Let

<img src="https://latex.codecogs.com/gif.latex?L&space;=&space;E_q[logp(x,z)-logq(z|\lambda)]" title="L = E_q[logp(x,z)-logq(z|\lambda)]" />

Then,
<!--\frac{\partial}{\partial \lambda}L = \frac{\partial}{\partial \lambda} \int[logp(x,z)-logq(z|\lambda)]q(z|\lambda)dz \\
=\int \frac{\partial}{\partial \lambda}[logp(x,z)-logq(z|\lambda)]q(z|\lambda)dz+\\
\int \frac{\partial}{\partial \lambda}q(z|\lambda)[logp(x,z)-logq(z|\lambda)]dz\\
=-\frac{\partial}{\partial \lambda}\int q(z|\lambda)dz +\int \frac{\partial}{\partial \lambda}logq(z|\lambda)[logp(x,z)-logq(z|\lambda)]q(z|\lambda)dz\\
=E_q[\frac{\partial}{\partial \lambda}logq(z|\lambda)[logp(x,z)-logq(z|\lambda)]]-->
<img src="https://latex.codecogs.com/gif.latex?\frac{\partial}{\partial&space;\lambda}L&space;=&space;\frac{\partial}{\partial&space;\lambda}&space;\int[logp(x,z)-logq(z|\lambda)]q(z|\lambda)dz&space;\\&space;=\int&space;\frac{\partial}{\partial&space;\lambda}[logp(x,z)-logq(z|\lambda)]q(z|\lambda)dz&plus;\\&space;\int&space;\frac{\partial}{\partial&space;\lambda}q(z|\lambda)[logp(x,z)-logq(z|\lambda)]dz\\&space;=-\frac{\partial}{\partial&space;\lambda}\int&space;q(z|\lambda)dz&space;&plus;\int&space;\frac{\partial}{\partial&space;\lambda}logq(z|\lambda)[logp(x,z)-logq(z|\lambda)]q(z|\lambda)dz\\&space;=E_q[\frac{\partial}{\partial&space;\lambda}logq(z|\lambda)[logp(x,z)-logq(z|\lambda)]]" title="\frac{\partial}{\partial \lambda}L = \frac{\partial}{\partial \lambda} \int[logp(x,z)-logq(z|\lambda)]q(z|\lambda)dz \\ =\int \frac{\partial}{\partial \lambda}[logp(x,z)-logq(z|\lambda)]q(z|\lambda)dz+\\ \int \frac{\partial}{\partial \lambda}q(z|\lambda)[logp(x,z)-logq(z|\lambda)]dz\\ =-\frac{\partial}{\partial \lambda}\int q(z|\lambda)dz +\int \frac{\partial}{\partial \lambda}logq(z|\lambda)[logp(x,z)-logq(z|\lambda)]q(z|\lambda)dz\\ =E_q[\frac{\partial}{\partial \lambda}logq(z|\lambda)[logp(x,z)-logq(z|\lambda)]]" />

The above results gives us a noisy unbiased gradient approximation using Monte Carlo samples from the variational distribution.
<!--\frac {1}{S} \sum_s \frac{\partial}{\partial \lambda}logq(z_s|\lambda)[logp(x,z_s)-logq(z_s|\lambda)],
z_s\sim q(z|\lambda)-->

<img src="https://latex.codecogs.com/gif.latex?\frac&space;{1}{S}&space;\sum_s&space;\frac{\partial}{\partial&space;\lambda}logq(z_s|\lambda)[logp(x,z_s)-logq(z_s|\lambda)],&space;z_s\sim&space;q(z|\lambda)" title="\frac {1}{S} \sum_s \frac{\partial}{\partial \lambda}logq(z_s|\lambda)[logp(x,z_s)-logq(z_s|\lambda)], z_s\sim q(z|\lambda)" />

The logq is called the score function. Now, we can optimize the ELBO by stocastic optimization. The optimizing only concern the variational distribution, with no assumption 
about the model.
However, the variance may be too large and slow down the optimization (we know that it is better to keep a low learning rate when the 
variance is large). Many methods have been proposed to control the variance of sampling from the variational distribution. The reference 
proves examples.

Variational inference is generalization of variation optimization, with the inclusion of latent variables.

## Conclusion
This post concerns the sampling nature of variation inference and the bound for maximizing the likelihood of continues or discrete 
latent variable models. It is interesting to connect EM algorithm, variational optimization, variational inference and variational 
autoencoder. 

In variation inference, we optimize the KL distance between P(x,z) and q(z) instead of P(z|x) in the ELBO. To be exact, KL(q(x)||p(z|x)) (KL(P(z|x)||q(z)) is not easy to take expectaion, known as expectation propagation) ensures to minimize the error in the domain of q, making q a tight distribution. We are confident in high q regions because of the expectation. But marginal q(z) is smaller than P(z).

Reference

> Ranganath R, Gerrish S, Blei D. Black box variational inference[C]//Artificial Intelligence and Statistics. 2014: 814-822.

> http://blog.csdn.net/jackytintin/article/details/53641885

> http://blog.otoro.net/2016/04/01/generating-large-images-from-latent-vectors/

> https://jmetzen.github.io/2015-11-27/vae.html

> Fox C W, Roberts S J. A tutorial on variational Bayesian inference[J]. Artificial intelligence review, 2012: 1-11.

> [Variational Inference tutorial by David Blei] (https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/variational-inference-i.pdf)
