## Logit Loss

Logistic probabilistc distribution function (PDF) is:
<!--F(x)=\frac{1}{1+e^{-\frac{(x-\mu)}{\gamma }}}-->
<a href="https://www.codecogs.com/eqnedit.php?latex=F(x)=\frac{1}{1&plus;e^{-\frac{(x-\mu)}{\gamma&space;}}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?F(x)=\frac{1}{1&plus;e^{-\frac{(x-\mu)}{\gamma&space;}}}" title="F(x)=\frac{1}{1+e^{-\frac{(x-\mu)}{\gamma }}}" /></a>
where <a href="https://www.codecogs.com/eqnedit.php?latex=\mu" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu" title="\mu" /></a>
and <a href="https://www.codecogs.com/eqnedit.php?latex=\gamma" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\gamma" title="\gamma" /></a> controls
the location and shape of the distribuion.

Logit loss is:
<!--log(\frac{1}{P(y|x)})=log(1+exp(-yf(x)))-->
<a href="https://www.codecogs.com/eqnedit.php?latex=log(\frac{1}{P(y|x)})=log(1&plus;exp(-yf(x)))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?log(\frac{1}{P(y|x)})=log(1&plus;exp(-yf(x)))" title="log(\frac{1}{P(y|x)})=log(1+exp(-yf(x)))" /></a>
