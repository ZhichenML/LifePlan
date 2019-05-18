% Variational Optimisation
% f(x) is a simple non-differential function
% p(x|theta) is a Gaussian
% increase learning rate, decrease sample size, will increase bias
% the value of f influce the strength of the gradients. I increase the
% function value to prevent zeros, then the algorithm run well.
% The algorithm is sensitive to the initial value and number of iterations,
% and may not converge
% Large sd may help convergence but may end up finding local optima.
% suggestion: set large sd, sample size, adjust learning rate
clear all;

% randn('seed',5);
D = 2; % dimension of parameter
% Create a nondifferencial loss function
w1 = linspace(1,2*pi,50);  w2 = w1;
for i = 1: length(w1)
    for j = 1:length(w2)
        Eval(i,j) = non_diff_E([w1(i),w2(j)]);
    end
end
% plot the error surface:
h=surf(w1,w2,Eval); set(h,'LineStyle','none');  view(0,90); hold on


Winit=[4 4]; % initial starting point for the optimisation

% Variational Optimisation:
Nsamples=100; % number of samples
sd=10; % initial standard deviation of the Gaussian
beta=2*log(sd); % parameterise the standard variance
mu=Winit; % initial mean of the Gaussian
sdvals=[sd];
betas = [beta];
eta = 0.1;
Nloops =50;
for i=1:Nloops
%     scatter3(mu(2),mu(1),non_diff_E(mu)+0.1,40,i);
    plot3(mu(2),mu(1),non_diff_E(mu)+0.1,'r.','markersize',20);%
    xsample=repmat(mu,Nsamples,1)+sd*randn(Nsamples,D); % draw samples
    
    g=zeros(1,D); % initialise the gradient for the mean mu
    gbeta=0; % initialise the gradient for the standard deviation (beta par)
    for j=1:Nsamples
        f(j) = non_diff_E(xsample(j,:)); % function value (error)
        g=g+(xsample(j,:)-mu).*f(j)./(sd*sd);
%         sd = sd - 1/sd + sum((xsample(j,:)-mu),2)/(sd*sd*sd); % beta prevent numerical issues
        gbeta=gbeta+0.5*f(j)*(1+exp(-beta)*sum((xsample(j,:)-mu).^2))-D; % Why positive? why -D?
    end
    g = g./Nsamples;
    gbeta=gbeta/Nsamples;
    
    mu=mu-eta*g; % Stochastic gradient descent for the mean
    beta=beta-0.01*gbeta; % Stochastic gradient descent for the variance par
    % comment the line above to turn off variance adaptation
        
    sd=sqrt(exp(beta)); 
    
    betas = [betas;beta];
    sdvals=[sdvals sd];
end

figure; plot(sdvals);
figure; plot(betas)
