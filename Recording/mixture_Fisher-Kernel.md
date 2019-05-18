# mixture_Fisher-Kernel
This manuscript contains implementation of mixture of Fisher kernel.

Fisher kernel is to approximate data by a generative model. Then the original data is represented by the Fisher Score, which is defined as the gradient of loa likelihood function. The kernel is defined by the infomation metric, but in implementation, people usually ignore the information metrc and project the Fisher Score into a Euclidean space. It is assumed that the manifold spanned by the specific generative model is able to provide proper representation for the data.

However, the assumption of approximating the data with one generative model is too restrict and may not generalize to all the data.
Therefore, we propose mixture Fisher kernel to provide more powerful manifold representations for the original data.

The code will be available soon.
