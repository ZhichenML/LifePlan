## Abstract
Supervised Machine learning is all about designing an error surface and an optimization approach.
This post is going to talk about the mechanism and behaviour of two important optimization algorithms, Steepest gradient descent (SGD) and conjugate gradient descent (CG).  We restrict our attention to a simple convex quadratic optimization problem as a runtime example. The convergence and geometrical interpretation will be provided. The implementation demonstrates that CG converges with fewer iterations than SGD, but each iteration requires a little more time cost.

## Steepest Gradient Descent
Consider an error surface with a quadratic form <!--f(x)=\frac{1}{2}x^TAx-bx+c-->:
<a href="https://www.codecogs.com/eqnedit.php?latex=f(x)=\frac{1}{2}x^TAx-bx&plus;c" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)=\frac{1}{2}x^TAx-bx&plus;c" title="f(x)=\frac{1}{2}x^TAx-bx+c" /></a>

The curvature of the surface is mainly governed by <a href="https://www.codecogs.com/eqnedit.php?latex=A" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A" title="A" /></a>.  Suppose <a href="https://www.codecogs.com/eqnedit.php?latex=A" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A" title="A" /></a> is symmetric, positive definite and sparse (otherwise factor it), we are able to find the minima (maxima) readily with iterative methods. 

To find the minima, make the derivative of the quadratic function equal zero, we get <!--Ax=b-->
<a href="https://www.codecogs.com/eqnedit.php?latex=Ax=b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Ax=b" title="Ax=b" /></a>.

Suppose the optimal solution is <a href="https://www.codecogs.com/eqnedit.php?latex=x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x" title="x" /></a>, then we have:
<!--f(p)=f(x)+(p-x)^TA(p-x)-->
<a href="https://www.codecogs.com/eqnedit.php?latex=f(p)=f(x)&plus;(p-x)^TA(p-x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(p)=f(x)&plus;(p-x)^TA(p-x)" title="f(p)=f(x)+(p-x)^TA(p-x)" /></a>. 

This is because
<!--f(x+e) = \frac{1}{2}(x+e)^TA(x+e)-b(x+e)+c\\
=\frac{1}{2}x^TAx+\frac{1}{2}e^TAe+x^TAe-bx-be+c\\
=f(x)+\frac{1}{2}e^TAe-->
<a href="https://www.codecogs.com/eqnedit.php?latex=f(x&plus;e)&space;=&space;\frac{1}{2}(x&plus;e)^TA(x&plus;e)-b(x&plus;e)&plus;c\\&space;=\frac{1}{2}x^TAx&plus;\frac{1}{2}e^TAe&plus;x^TAe-bx-be&plus;c\\&space;=f(x)&plus;\frac{1}{2}e^TAe" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x&plus;e)&space;=&space;\frac{1}{2}(x&plus;e)^TA(x&plus;e)-b(x&plus;e)&plus;c\\&space;=\frac{1}{2}x^TAx&plus;\frac{1}{2}e^TAe&plus;x^TAe-bx-be&plus;c\\&space;=f(x)&plus;\frac{1}{2}e^TAe" title="f(x+e) = \frac{1}{2}(x+e)^TA(x+e)-b(x+e)+c\\ =\frac{1}{2}x^TAx+\frac{1}{2}e^TAe+x^TAe-bx-be+c\\ =f(x)+\frac{1}{2}e^TAe" /></a>

Hence, since  A is positive definite, f(x) is guaranteed to be the minimum. This can be intuitively understood by the fact that the shape of the function f(x) is a bow, which must be spanned by all the dimensions.

For each iteration <a href="https://www.codecogs.com/eqnedit.php?latex=i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?i" title="i" /></a>, let the error between the current solution and the optimal solution be 
<!--e_{(i)}=x_{(i)}-x-->
<a href="https://www.codecogs.com/eqnedit.php?latex=e_{(i)}=x_{(i)}-x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?e_{(i)}=x_{(i)}-x" title="e_{(i)}=x_{(i)}-x" /></a>

 To optimize the function, we need to go in the opposite direction of the gradient, that is  
<!----f'(x_{(i)}) = b-Ax_{(i)}=-A(x_{(i)}-x)=-Ae_{(i)}=r_{(i)}-->
<a href="https://www.codecogs.com/eqnedit.php?latex=-f'(x_{(i)})&space;=&space;b-Ax_{(i)}=-A(x_{(i)}-x)=-Ae_{(i)}=r_{(i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?-f'(x_{(i)})&space;=&space;b-Ax_{(i)}=-A(x_{(i)}-x)=-Ae_{(i)}=r_{(i)}" title="-f'(x_{(i)}) = b-Ax_{(i)}=-A(x_{(i)}-x)=-Ae_{(i)}=r_{(i)}" /></a>
i.e. the residual is equivalent to the steepest descent direction.

Having got  <!--r_{(i)}--> 
<a href="https://www.codecogs.com/eqnedit.php?latex=r_{(i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r_{(i)}" title="r_{(i)}" /></a>, we are able to update the solution by:
<!--x_{(i+1)}=x_{(i)}+\alpha_{(i)} r_{(i)}-->
<a href="https://www.codecogs.com/eqnedit.php?latex=x_{(i&plus;1)}=x_{(i)}&plus;\alpha_{(i)}&space;r_{(i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_{(i&plus;1)}=x_{(i)}&plus;\alpha_{(i)}&space;r_{(i)}" title="x_{(i+1)}=x_{(i)}+\alpha_{(i)} r_{(i)}" /></a>

How to determine the step size <!--\alpha_{(i)} --> <a href="https://www.codecogs.com/eqnedit.php?latex=\alpha_{(i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha_{(i)}" title="\alpha_{(i)}" /></a>?
The strategy is to think about if we already have <!--x_{(i+1)}-->
<a href="https://www.codecogs.com/eqnedit.php?latex=x_{(i&plus;1)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_{(i&plus;1)}" title="x_{(i+1)}" /></a>, then the only parameter for optimizing the original quadratic funtion is to find 
<!--\alpha_{(i)} --> <a href="https://www.codecogs.com/eqnedit.php?latex=\alpha_{(i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha_{(i)}" title="\alpha_{(i)}" /></a>?, whch can be achieved by setting the derivative w.r.t. <!--\alpha_{(i)} --> <a href="https://www.codecogs.com/eqnedit.php?latex=\alpha_{(i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\alpha_{(i)}" title="\alpha_{(i)}" /></a>?:
<!--\frac{df(x_{(i)})}{d\alpha_{(i)}}=f'(x_{(i+1)})^T \frac{dx_{(i+1)}}{d\alpha_{(i)}}=f'(x_{(i+1)})^Tr_{(i)}-->
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{df(x_{(i)})}{d\alpha_{(i)}}=f'(x_{(i&plus;1)})^T&space;\frac{dx_{(i&plus;1)}}{d\alpha_{(i)}}=f'(x_{(i&plus;1)})^Tr_{(i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{df(x_{(i)})}{d\alpha_{(i)}}=f'(x_{(i&plus;1)})^T&space;\frac{dx_{(i&plus;1)}}{d\alpha_{(i)}}=f'(x_{(i&plus;1)})^Tr_{(i)}" title="\frac{df(x_{(i)})}{d\alpha_{(i)}}=f'(x_{(i+1)})^T \frac{dx_{(i+1)}}{d\alpha_{(i)}}=f'(x_{(i+1)})^Tr_{(i)}" /></a>

The quation means the the next residual is orthogoal to the current residual, so that in each step, the current direction has no contribution to improve the solution for the next step. This is greedy. However, it does not mean we do not need this direction in the future. Actually, this is just the truth, because for the step after the next step, the orthogonal direction is again the current one. So the algorithm switches between two orthogonal directions.

Expand the equation, we have:
<!--f'(x_{(i+1)})^Tr_{(i)}\\
=(Ax_{(i+1)}-b)r_{(i)}\\
=(A(x_{(i)}+\alpha_{(i)}r_{(i)})-b)r_{(i)}\\
=(Ax_{(i)}-b)^Tr_{(i)}+\alpha_{(i)}r_{(i)}^TAr_{(i)}\\
=-r_{(i)}^Tr_{(i)}+\alpha_{(i)}r_{(i)}^TAr_{(i)}=0-->

<a href="https://www.codecogs.com/eqnedit.php?latex=f'(x_{(i&plus;1)})^Tr_{(i)}\\&space;=(Ax_{(i&plus;1)}-b)r_{(i)}\\&space;=(A(x_{(i)}&plus;\alpha_{(i)}r_{(i)})-b)r_{(i)}\\&space;=(Ax_{(i)}-b)^Tr_{(i)}&plus;\alpha_{(i)}r_{(i)}^TAr_{(i)}\\&space;=-r_{(i)}^Tr_{(i)}&plus;\alpha_{(i)}r_{(i)}^TAr_{(i)}=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f'(x_{(i&plus;1)})^Tr_{(i)}\\&space;=(Ax_{(i&plus;1)}-b)r_{(i)}\\&space;=(A(x_{(i)}&plus;\alpha_{(i)}r_{(i)})-b)r_{(i)}\\&space;=(Ax_{(i)}-b)^Tr_{(i)}&plus;\alpha_{(i)}r_{(i)}^TAr_{(i)}\\&space;=-r_{(i)}^Tr_{(i)}&plus;\alpha_{(i)}r_{(i)}^TAr_{(i)}=0" title="f'(x_{(i+1)})^Tr_{(i)}\\ =(Ax_{(i+1)}-b)r_{(i)}\\ =(A(x_{(i)}+\alpha_{(i)}r_{(i)})-b)r_{(i)}\\ =(Ax_{(i)}-b)^Tr_{(i)}+\alpha_{(i)}r_{(i)}^TAr_{(i)}\\ =-r_{(i)}^Tr_{(i)}+\alpha_{(i)}r_{(i)}^TAr_{(i)}=0" /></a>

Then,
<!--\\
r_{(i)}^Tr_{(i)}=\alpha_{(i)}r_{(i)}^TAr_{(i)}\\
\alpha_{(i)}=\frac{r_{(i)}^Tr_{(i)}}{r_{(i)}^TAr_{(i)}}-->

<a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;r_{(i)}^Tr_{(i)}=\alpha_{(i)}r_{(i)}^TAr_{(i)}\\&space;\alpha_{(i)}=\frac{r_{(i)}^Tr_{(i)}}{r_{(i)}^TAr_{(i)}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;r_{(i)}^Tr_{(i)}=\alpha_{(i)}r_{(i)}^TAr_{(i)}\\&space;\alpha_{(i)}=\frac{r_{(i)}^Tr_{(i)}}{r_{(i)}^TAr_{(i)}}" title="\\ r_{(i)}^Tr_{(i)}=\alpha_{(i)}r_{(i)}^TAr_{(i)}\\ \alpha_{(i)}=\frac{r_{(i)}^Tr_{(i)}}{r_{(i)}^TAr_{(i)}}" /></a>

To this end, we have the optimization direction and the step size. The steepest descent algorithm can be summarized as follows:
<!--\\
r_{(0)} = b - A x_{(0)}\\
\alpha_{(i)}=\frac{r_{(i)}^Tr_{(i)}}{r_{(i)}^TAr_{(i)}}, (i=0,1,\cdots)\\
x_{(i+1)} = x_{(i)} + \alpha_{(i)}r_{(i)}\\
(\text{or,  } r_{(i+1)} = r_{(i)} - \alpha_{(i)}Ar_{(i)})-->

<a href="https://www.codecogs.com/eqnedit.php?latex=\\&space;r_{(0)}&space;=&space;b&space;-&space;A&space;x_{(0)}\\&space;\alpha_{(i)}=\frac{r_{(i)}^Tr_{(i)}}{r_{(i)}^TAr_{(i)}},&space;(i=0,1,\cdots)\\&space;x_{(i&plus;1)}&space;=&space;x_{(i)}&space;&plus;&space;\alpha_{(i)}r_{(i)}\\&space;(\text{or,&space;}&space;r_{(i&plus;1)}&space;=&space;r_{(i)}&space;-&space;\alpha_{(i)}Ar_{(i)})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\\&space;r_{(0)}&space;=&space;b&space;-&space;A&space;x_{(0)}\\&space;\alpha_{(i)}=\frac{r_{(i)}^Tr_{(i)}}{r_{(i)}^TAr_{(i)}},&space;(i=0,1,\cdots)\\&space;x_{(i&plus;1)}&space;=&space;x_{(i)}&space;&plus;&space;\alpha_{(i)}r_{(i)}\\&space;(\text{or,&space;}&space;r_{(i&plus;1)}&space;=&space;r_{(i)}&space;-&space;\alpha_{(i)}Ar_{(i)})" title="\\ r_{(0)} = b - A x_{(0)}\\ \alpha_{(i)}=\frac{r_{(i)}^Tr_{(i)}}{r_{(i)}^TAr_{(i)}}, (i=0,1,\cdots)\\ x_{(i+1)} = x_{(i)} + \alpha_{(i)}r_{(i)}\\ (\text{or, } r_{(i+1)} = r_{(i)} - \alpha_{(i)}Ar_{(i)})" /></a>

For the last equation, by replacing the solution updating by the residual updating, the computational cost could be reduce by omiting <!--A x_{(i)}--> <a href="https://www.codecogs.com/eqnedit.php?latex=A&space;x_{(i)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?A&space;x_{(i)}" title="A x_{(i)}" /></a>. The first equation need to be periodical used to preserve calculation precision.

Note that 
<!--\frac{r^Tr}{r^TAr} \in [\frac{1}{\lambda_{max}},\frac{1}{\lambda_{min}}]-->

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{r^Tr}{r^TAr}&space;\in&space;[\frac{1}{\lambda_{max}},\frac{1}{\lambda_{min}}]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{r^Tr}{r^TAr}&space;\in&space;[\frac{1}{\lambda_{max}},\frac{1}{\lambda_{min}}]" title="\frac{r^Tr}{r^TAr} \in [\frac{1}{\lambda_{max}},\frac{1}{\lambda_{min}}]" /></a>.

This is because the search direction could be viewed by the linear combination of the eigenvectors the matrix in the quadratic term(Symmetric matrix has eigenvectors that are able to span the whole space). By substituting the equation 
<!--Ar = \lambda r-->

<a href="https://www.codecogs.com/eqnedit.php?latex=Ar&space;=&space;\lambda&space;r" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Ar&space;=&space;\lambda&space;r" title="Ar = \lambda r" /></a>
The result is immediately obtained.

If the condition number  <!-- k =\lambda_{max}/ \lambda_{min}--> <a href="https://www.codecogs.com/eqnedit.php?latex=k&space;=\lambda_{max}/&space;\lambda_{min}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k&space;=\lambda_{max}/&space;\lambda_{min}" title="k =\lambda_{max}/ \lambda_{min}" /></a> is large, the optimization problem become ill-conditioned optimization. The convergence rate is propotional to <!--\frac{k-1}{k+1}-->

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{k-1}{k&plus;1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{k-1}{k&plus;1}" title="\frac{k-1}{k+1}" /></a>.

In this case, the convergence rate is sensitive to the initial solution. The situation would be better is the initial solution is on or near the eigenvector of 
<\lambda_{max}>
<a href="https://www.codecogs.com/eqnedit.php?latex=\lambda_{max}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\lambda_{max}" title="\lambda_{max}" /></a>.

When the condition number is small, it is not sensitive to the initial solution. That's desired A.

The spectral radius of A also controls the convergence rate. If the max eigenvalue is less then one, the error would diminish. The smaller the better, because <!--B^nv =\lambda^nv-->
<a href="https://www.codecogs.com/eqnedit.php?latex=B^nv&space;=\lambda^nv" target="_blank"><img src="https://latex.codecogs.com/gif.latex?B^nv&space;=\lambda^nv" title="B^nv =\lambda^nv" /></a>. This is especially important for iterative methods, such Jacobi methods, which reparametrize the A so that only iterative production is needed.

Congugate Gradient descent is to find a set of congugate directions with respect to A, such that each direction only involves once in the 
optimization. (could say more if possible...)




## Refference
1. > [An Introduction to the Conjugate Gradient Method Without the Agonizing Pain](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)

