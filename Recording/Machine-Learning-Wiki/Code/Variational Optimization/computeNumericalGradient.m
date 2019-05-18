function Numgrad = computeNumericalGradient(J, theta)


eps = 10^(-4);
Numgrad = zeros(size(theta));
for i = 1:numel(theta)
    up = theta; low = theta;
    up(i) = up(i) + eps; low(i) = low(i) - eps;
    Numgrad(i)  = (J(up)-J(low))/(2*eps);
end