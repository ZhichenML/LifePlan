function VO_turorial()
% visualization of the original variable space to parameter space of the
% distribution.

% differentiale
x = linspace(-2*pi,2*pi,1000);
y = -sinc(x)*5-10;
figure(1); plot(y);ylabel('f(x)'); xlabel('x');
u = linspace(-5,5,50);
sigma = linspace(0,5,50);

for i = 1:length(u)
    for j = 1:length(sigma)
        xsample = randn(500,1)*sigma(j) + repmat(u(i),500,1);
        Eval(i,j) = mean(-sinc(xsample)*5-10);
    end
end
figure(2);
surf(sigma,u,Eval); view(270,90); ylabel('u'); xlabel('sd'); zlabel('E[f(x)]');

pause();

% non-differentiable, sd, 
clear;clc;

x = linspace(-2*pi,2*pi,1000);
y = -round(sinc(x)*5)-10;
figure(1); plot(y);ylabel('f(x)'); xlabel('x');
u = linspace(-5,5,50);
sigma = linspace(0,5,50);

for i = 1:length(u)
    for j = 1:length(sigma)
        xsample = randn(500,1)*sigma(j) + repmat(u(i),500,1);
        Eval(i,j) = mean(-round(sinc(xsample)*5)-10);
    end
end
figure(2);
surf(sigma,u,Eval); view(270,90); xlabel('sd'); ylabel('u'); zlabel('E[f(x)]')

pause();
% fixed sigma
% The sd controls the bias/variance trade-off
clear;clc;
x = linspace(1,300,300);
for i = 1:length(x)
    y(i) = Eval_f(x(i));
end
figure(1); plot(y); xlabel('x'); ylabel('f(x)');

u = linspace(1,300,300);
sigma = linspace(0,50,50);

for i = 1:length(u)
    for j = 1:length(sigma)
        xsample = randn(500,1)*sigma(j) + repmat(u(i),500,1);
        for ind = 1:500
            tmp(ind) = Eval_f(xsample(ind));
        end
        Eval(i,j) = mean(tmp);
    end
end
figure(2);
surf(sigma,u,Eval); view(270,90); ylabel('sd'); xlabel('u'); zlabel('E[f(x)]')
figure(3);
plot(Eval(:,10)); xlabel('x'); ylabel('E[f(x)]')
end

function v = Eval_f(x)
%x = [zeros(50,1);cos(linspace(1,pi,100))';zeros(50,1);cos(linspace(1,pi,50))'*1.1];
tmp = linspace(1,pi,100)';
if (x<=50) v = 0;
elseif (x>=51 && x<=150) v = cos(pi*(x-50)/100);
elseif (x>=151 && x<=200) v = 0;
elseif (x>=201 && x<=250) v = cos(pi*(x-200)/50)*1.1;
else v = 0;
end
end