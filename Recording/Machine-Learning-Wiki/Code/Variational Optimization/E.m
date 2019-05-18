function out=E(W,x,y)
% loss function
out=mean(0.5*(W*x-y).^2);
% out = log(1+exp(-W*x*y'));