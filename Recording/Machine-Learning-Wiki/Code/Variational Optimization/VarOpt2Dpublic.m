% Variational Optimisation
% f(x) is a simple quadratic objective function (linear regression sq loss)
% p(x|theta) is a Gaussian

clear all;

randn('seed',5);

% Create the dataset:
N=100; % Number of datapoints
D=2; % Dimension of the data
W0=randn(1,D)/sqrt(D); % true linear regression weight
x=randn(D,N); % inputs
y=W0*x; % outputs

% plot the error surface:
w1=linspace(-5,5,100); w2=w1;
for i=1:length(w1)
    for j=1:length(w2)
        Esurf(i,j)=E([w1(i) w2(j)],x,y);
    end
end
h=surf(w1,w2,Esurf); set(h,'LineStyle','none');  view(0,90); hold on


Winit=[-4 4]; % initial starting point for the optimisation

% standard gradient descent:
Nloops=50; % number of iterations
eta=0.1; % learning rate
W=Winit;
for i=1:Nloops
    plot3(W(2),W(1),E(W,x,y)+0.1,'y.','markersize',20);
    W=W-eta*gradE(W,x,y);
end

% Variational Optimisation:
Nsamples=10; % number of samples
sd=5; % initial standard deviation of the Gaussian
beta=2*log(sd); % parameterise the standard variance
mu=Winit; % initial mean of the Gaussian
sdvals=[sd];
betas = [beta];
for i=1:Nloops
    plot3(mu(2),mu(1),E(mu,x,y)+0.1,'r.','markersize',20);%
    EvalVarOpt(i)=E(mu,x,y); % error value
    xsample=repmat(mu,Nsamples,1)+sd*randn(Nsamples,D); % draw samples
    
    g=zeros(1,D); % initialise the gradient for the mean mu
    gbeta=0; % initialise the gradient for the standard deviation (beta par)
    for j=1:Nsamples
        f(j) = E(xsample(j,:),x,y); % function value (error)
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