function [cost, grad] = exam(x)
% f = 2x_1x_2 + 2x_1
cost = 2*x(1)*x(2)+2*x(1);

grad(1) = 2*x(2)+2;
grad(2) = 2*x(1);

end

function demo()
[cost, grad] = exam([1,2]);
numgrad = Numgrad(@(x)exam(x),[1,2]);
end