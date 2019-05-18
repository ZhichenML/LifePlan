## Mutual information
We want to maximize the information retatined in <img src="https://latex.codecogs.com/gif.latex?y" title="y" /> about 
<img src="https://latex.codecogs.com/gif.latex?x" title="x" />, after cetertain transformations.

Mutual information is to minimize the distance between two distributions. Consider <img src="https://latex.codecogs.com/gif.latex?x" title="x" /> and <img src="https://latex.codecogs.com/gif.latex?y" title="y" /> with joint distribution <img src="https://latex.codecogs.com/gif.latex?p(x,y)" title="p(x,y)" />.

<!--I(x,y)=KL(p(x,y)||p(x)p(y))\\
=E_{p(x,y)}[-logp(y)+logp(y|x)]\\
=H[y]+E_{p(x,y)}[log q_\theta(y|x)\frac{p(y|x)}{q_\theta(y|x)}]\\
=H[y]+E_{p(x,y)}[log q_\theta(y|x))]+KL(p(y|x)||q_\theta(y|x))\\
\geq H[y]+E_{p(x,y)}[log q_\theta(y|x))]-->

<img src="https://latex.codecogs.com/gif.latex?I(x,y)=KL(p(x,y)||p(x)p(y))\\&space;=E_{p(x,y)}[-logp(y)&plus;logp(y|x)]\\&space;=H[y]&plus;E_{p(x,y)}[log&space;q_\theta(y|x)\frac{p(y|x)}{q_\theta(y|x)}]\\&space;=H[y]&plus;E_{p(x,y)}[log&space;q_\theta(y|x))]&plus;KL(p(y|x)||q_\theta(y|x))\\&space;\geq&space;H[y]&plus;E_{p(x,y)}[log&space;q_\theta(y|x))]" title="I(x,y)=KL(p(x,y)||p(x)p(y))\\ =E_{p(x,y)}[-logp(y)+logp(y|x)]\\ =H[y]+E_{p(x,y)}[log q_\theta(y|x)\frac{p(y|x)}{q_\theta(y|x)}]\\ =H[y]+E_{p(x,y)}[log q_\theta(y|x))]+KL(p(y|x)||q_\theta(y|x))\\ \geq H[y]+E_{p(x,y)}[log q_\theta(y|x))]" />

In [this paper](http://aivalley.com/Papers/MI_NIPS_final.pdf), the variational distribution is linked to the generative model.

Consider the generative model in GAN is parameterized by <img src="https://latex.codecogs.com/gif.latex?\theta" title="\theta" />.
Then 
<!--l(\theta)\geq H(y)+E_{p(x,y)}[logq_\theta(y|x)]\\
=-log(0.5) +E_{p(x,y)}[logq_\theta(1|x_{real})+logq_\theta(0|x_{fake})]-->
<img src="https://latex.codecogs.com/gif.latex?l(\theta)\geq&space;H(y)&plus;E_{p(x,y)}[logq_\theta(y|x)]\\&space;=-log(0.5)&space;&plus;E_{p(x,y)}[logq_\theta(1|x_{real})&plus;logq_\theta(0|x_{fake})]" title="l(\theta)\geq H(y)+E_{p(x,y)}[logq_\theta(y|x)]\\ =-log(0.5) +E_{p(x,y)}[logq_\theta(1|x_{real})+logq_\theta(0|x_{fake})]" />

<!--l(\theta)-log(2) \geq  E_{p(x,y)}[logq_\theta(1|x_{real})+logq_\theta(0|x_{fake})]\\
=E_{x_{real}}[logD(x)]+E_{x_{fake},c}[log(1-D(x))]-->
<img src="https://latex.codecogs.com/gif.latex?l(\theta)-log(2)&space;\geq&space;E_{p(x,y)}[logq_\theta(1|x_{real})&plus;logq_\theta(0|x_{fake})]\\&space;=E_{x_{real}}[logD(x)]&plus;E_{x_{fake},c}[log(1-D(x))]" title="l(\theta)-log(2) \geq E_{p(x,y)}[logq_\theta(1|x_{real})+logq_\theta(0|x_{fake})]\\ =E_{x_{real}}[logD(x)]+E_{x_{fake},c}[log(1-D(x))]" />


## Relation to Autoencoder, 
which tries to reconstruct <img src="https://latex.codecogs.com/gif.latex?x" title="x" />.
<!--log P(\hat{x}|x)=log\int_y P(\hat{x}|y)P(y|x)dy \geq \int_y P(y|x)log P(\hat{x}=s|y)dy\\
\int_x P(x)logP(\hat{x}|x)\geq \int_x P(x)\int_yP(y|x) log P(\hat{x}|y)=\int_{x,y}P(x,y) log Q(x|y)=E_{P(x,y)}[Q(x|y)]-->
<a href="https://www.codecogs.com/eqnedit.php?latex=log&space;P(\hat{x}|x)=log\int_y&space;P(\hat{x}|y)P(y|x)dy&space;\geq&space;\int_y&space;P(y|x)log&space;P(\hat{x}=s|y)dy\\&space;\int_x&space;P(x)logP(\hat{x}|x)\geq&space;\int_x&space;P(x)\int_yP(y|x)&space;log&space;P(\hat{x}|y)=\int_{x,y}P(x,y)&space;log&space;Q(x|y)=E_{P(x,y)}[Q(x|y)]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?log&space;P(\hat{x}|x)=log\int_y&space;P(\hat{x}|y)P(y|x)dy&space;\geq&space;\int_y&space;P(y|x)log&space;P(\hat{x}=s|y)dy\\&space;\int_x&space;P(x)logP(\hat{x}|x)\geq&space;\int_x&space;P(x)\int_yP(y|x)&space;log&space;P(\hat{x}|y)=\int_{x,y}P(x,y)&space;log&space;Q(x|y)=E_{P(x,y)}[Q(x|y)]" title="log P(\hat{x}|x)=log\int_y P(\hat{x}|y)P(y|x)dy \geq \int_y P(y|x)log P(\hat{x}=s|y)dy\\ \int_x P(x)logP(\hat{x}|x)\geq \int_x P(x)\int_yP(y|x) log P(\hat{x}|y)=\int_{x,y}P(x,y) log Q(x|y)=E_{P(x,y)}[Q(x|y)]" /></a>

Therefore, reconstruction implicitly assures Multual information variational lower bound maximization. 












