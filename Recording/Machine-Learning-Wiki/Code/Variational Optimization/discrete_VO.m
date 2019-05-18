% Variational Optimisation
% f(x) is a 1 dimensional non-differential with discrete variables.
% p(x|theta) is a Gaussian

clear all;

randn('seed',1);

% Create the dataset:

D=1; % Dimension of the data

% plot the error surface:
Eval = [ones(100,1)*10;ones(200,1)*4;ones(100,1)*8;ones(300,1)*9;ones(100,1)*5;ones(200,1)*1];

plot(Eval);   hold on


% Winit=round(rand()*1000); % initial starting point for the optimisation
Winit = 4;

% standard gradient descent:
Nloops=150; % number of iterations
eta=0.1; % learning rate
W=Winit;

% Variational Optimisation:
Nsamples=50; % number of samples
sd=300; % initial standard deviation of the Gaussian
beta=2*log(sd); % parameterise the standard variance
mu = Winit; % initial mean of the Gaussian
sdvals=[sd];
betas = [beta];
for i=1:Nloops
    plot(mu,Eval(mu)+0.1,'r.','markersize',20);%
    EvalVarOpt(i)=Eval(mu); % error value
    for ind = 1:Nsamples
        success = 0;
        while success == 0
            xsample(ind) = mu + round(sd*randn());
            if xsample(ind) <= 0 || xsample(ind) >1000
                success = 0;
            else
                success = 1;
            end
        end
    end
    %xsample=repmat(mu,Nsamples,1)+round(sd*randn(Nsamples,D)); % draw samples
    
    g=zeros(1,D); % initialise the gradient for the mean mu
    gbeta=0; % initialise the gradient for the standard deviation (beta par)
    for j=1:Nsamples
        f(j) = Eval(xsample(j)); % function value (error)
        g=g+(xsample(j)-mu).*f(j)./(sd*sd);
%         sd = sd - 1/sd + sum((xsample(j,:)-mu),2)/(sd*sd*sd); % beta prevent numerical issues
        gbeta=gbeta+0.5*f(j)*(1+exp(-beta)*sum((xsample(j)-mu).^2))-D; % Why positive? why -D?
    end
    g = g./Nsamples;
    gbeta=gbeta/Nsamples;
    
    mu=mu-round(sd*10*g) % Stochastic gradient descent for the mean
    success = 0;
    while success ==0
    if mu <= 0 
        mu = mu+round(sd*10*g)+round(randn()*500);
    elseif  mu > 1000
        mu = mu+round(sd*10*g)+round(randn()*500);
    end
    if mu>0 && mu < 1001
        success =1;
    end
    end
    beta=beta-0.01*gbeta; % Stochastic gradient descent for the variance par
    % comment the line above to turn off variance adaptation
        
    sd=sqrt(exp(beta)); 
    
    betas = [betas;beta];
    sdvals=[sdvals sd];
end

figure; plot(sdvals);
figure; plot(betas)
