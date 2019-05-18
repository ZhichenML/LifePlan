Logistic regreesion is a discrimination model.
I am going to introduce an interesting logistic regression for a generative purpose, being aware the order information of data.

Image we have data <img src="https://latex.codecogs.com/gif.latex?x=[x_1,x_2]" title="x=[x_1,x_2]" /> generated from distribution <img src="https://latex.codecogs.com/gif.latex?P(x)" title="P(x)" />.  We want to make the discrinator notice that <img src="https://latex.codecogs.com/gif.latex?[x_2,x_1]" title="[x_2,x_1]" /> is not right. For this purpose, we employ a binary logisticregression as the discrinator.
We approximate the <img src="https://latex.codecogs.com/gif.latex?P(x)" title="P(x)" /> by <img src="https://latex.codecogs.com/gif.latex?Q_\theta(x)" title="Q_\theta(x)" />. Then the loss function to be minimized for LR is:
<!--L(\theta)=-logP(x|theta) = -E_{x\sim P}[log\frac{Q_\theta(x)}{Q_\theta(x)+Q_\theta(\hat{x})}]\\
=-E_{x\sim P}[log\frac{Q_\theta(x)}{P(x)}P(x)]+E_{x\sim P}[Q_\theta(x)+Q_\theta(\hat{x})]\\
=KL[P(x)||Q_\theta(x)]-KL[P(x)||Q_\theta(x)+Q_\theta(\hat{x})]-->
<img src="https://latex.codecogs.com/gif.latex?L(\theta)=-logP(x|theta)&space;=&space;-E_{x\sim&space;P}[log\frac{Q_\theta(x)}{Q_\theta(x)&plus;Q_\theta(\hat{x})}]\\&space;=-E_{x\sim&space;P}[log\frac{Q_\theta(x)}{P(x)}P(x)]&plus;E_{x\sim&space;P}[Q_\theta(x)&plus;Q_\theta(\hat{x})]\\&space;=KL[P(x)||Q_\theta(x)]-KL[P(x)||Q_\theta(x)&plus;Q_\theta(\hat{x})]" title="L(\theta)=-logP(x|theta) = -E_{x\sim P}[log\frac{Q_\theta(x)}{Q_\theta(x)+Q_\theta(\hat{x})}]\\ =-E_{x\sim P}[log\frac{Q_\theta(x)}{P(x)}P(x)]+E_{x\sim P}[Q_\theta(x)+Q_\theta(\hat{x})]\\ =KL[P(x)||Q_\theta(x)]-KL[P(x)||Q_\theta(x)+Q_\theta(\hat{x})]" />

The minimization loss function now turns out to be minimizing two KL divergences. The first term means we should approximate the true data generative distribution faithfully. However, the second term is not clear.

Reorganize the second term, we obtain
<!--KL[P(x)||Q_\theta(x)+Q_\theta(\hat{x})]\\
=E_{x\sim P(x)}[\frac{P(x)}{Q_\theta(x)+Q_\theta(\hat{x})}]\\
=E_{x\sim P(x)}[\frac{P(x)}{P(x)+P(\hat{x})}\frac{P(x)+P(\hat{x})}{Q_\theta(x)+Q_\theta(\hat{x})}]\\
=KL[P(x)||P(x)+P(\hat{x})]+KL[P(x)+P(\hat{x})||Q_\theta(x)+Q_\theta(\hat{x})]-->
<img src="https://latex.codecogs.com/gif.latex?KL[P(x)||Q_\theta(x)&plus;Q_\theta(\hat{x})]\\&space;=E_{x\sim&space;P(x)}[\frac{P(x)}{Q_\theta(x)&plus;Q_\theta(\hat{x})}]\\&space;=E_{x\sim&space;P(x)}[\frac{P(x)}{P(x)&plus;P(\hat{x})}\frac{P(x)&plus;P(\hat{x})}{Q_\theta(x)&plus;Q_\theta(\hat{x})}]\\&space;=KL[P(x)||P(x)&plus;P(\hat{x})]&plus;KL[P(x)&plus;P(\hat{x})||Q_\theta(x)&plus;Q_\theta(\hat{x})]" title="KL[P(x)||Q_\theta(x)+Q_\theta(\hat{x})]\\ =E_{x\sim P(x)}[\frac{P(x)}{Q_\theta(x)+Q_\theta(\hat{x})}]\\ =E_{x\sim P(x)}[\frac{P(x)}{P(x)+P(\hat{x})}\frac{P(x)+P(\hat{x})}{Q_\theta(x)+Q_\theta(\hat{x})}]\\ =KL[P(x)||P(x)+P(\hat{x})]+KL[P(x)+P(\hat{x})||Q_\theta(x)+Q_\theta(\hat{x})]" />

Then the original loss function is:
<!--L(\theta)=-logP(x|theta) = -E_{x\sim P}[log\frac{Q_\theta(x)}{Q_\theta(x)+Q_\theta(\hat{x})}]\\
=-E_{x\sim P}[log\frac{Q_\theta(x)}{P(x)}P(x)]+E_{x\sim P}[Q_\theta(x)+Q_\theta(\hat{x})]\\
=KL[P(x)||Q_\theta(x)]-KL[P(x)+P(\hat{x})||Q_\theta(x)+Q_\theta(\hat{x})]-KL[P(x)||P(x)+P(\hat{x})]-->
<img src="https://latex.codecogs.com/gif.latex?L(\theta)=-logP(x|theta)&space;=&space;-E_{x\sim&space;P}[log\frac{Q_\theta(x)}{Q_\theta(x)&plus;Q_\theta(\hat{x})}]\\&space;=-E_{x\sim&space;P}[log\frac{Q_\theta(x)}{P(x)}P(x)]&plus;E_{x\sim&space;P}[Q_\theta(x)&plus;Q_\theta(\hat{x})]\\&space;=KL[P(x)||Q_\theta(x)]-KL[P(x)&plus;P(\hat{x})||Q_\theta(x)&plus;Q_\theta(\hat{x})]-KL[P(x)||P(x)&plus;P(\hat{x})]" title="L(\theta)=-logP(x|theta) = -E_{x\sim P}[log\frac{Q_\theta(x)}{Q_\theta(x)+Q_\theta(\hat{x})}]\\ =-E_{x\sim P}[log\frac{Q_\theta(x)}{P(x)}P(x)]+E_{x\sim P}[Q_\theta(x)+Q_\theta(\hat{x})]\\ =KL[P(x)||Q_\theta(x)]-KL[P(x)+P(\hat{x})||Q_\theta(x)+Q_\theta(\hat{x})]-KL[P(x)||P(x)+P(\hat{x})]" />

That indicates we need to maximize the distance between <img src="https://latex.codecogs.com/gif.latex?Q_\theta(x)&plus;Q_\theta(\hat{x})" title="Q_\theta(x)+Q_\theta(\hat{x})" /> and <img src="https://latex.codecogs.com/gif.latex?P(x)&plus;P(\hat{x})" title="P(x)+P(\hat{x})" />. Basically, when the generative distribution is approximated perfectly, the second term would be zero. Therefore, we `do not` want the data be modeled precisely, we also care the order of the data.

This observation is useful for data where the order is important for understanding the data, let alone the data modeling for typically generative purposes.


## Ref
Hyvarinen A, Morioka H. Unsupervised feature extraction by time-contrastive learning and nonlinear ICA[C]//Advances in Neural Information Processing Systems. 2016: 3765-3773.

Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles
