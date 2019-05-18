Logistic regression is a linear model for classification. It can also be viewed as a filter. The loss it employs is log-loss. The learning 
strategy is to find a hypeplane that makes data from different class stay away from the hypeplane as far as possible. When a sample is 
correctly classified, the log(out)(for class 1) or log(1-out)(for class 0) is added in the loss function. The objective is to maximize this 
summation over all samples.


This file consists of an implementation of Logistic regression. In detail, the amin steps include the K-fold data division, function that
returns the objective value and gradients. I made a mistakes when preparing the data, the idea is to use the random index as the index of
data index. One may easily confuses by leting the random index as the data index. The other mistake is when I calculate the gradients. Note
that you can off course use average loss and average gradient instead of full values, which may be too large. I also find meshgrid, bsxfun,
and function handle really handy and helpful.
