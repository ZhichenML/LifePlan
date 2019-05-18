function grad = Numgrad(J,theta)

npara = numel(theta);
eps = 10^(-4);
grad = zeros(npara,1);
for i=1:npara
    up = theta; up(i) = up(i)+eps;
    low = theta; low(i) = low(i) - eps;
    grad(i) = (J(up)-J(low))/(2*eps);
end

end

