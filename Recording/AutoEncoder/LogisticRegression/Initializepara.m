function theta = Initializepara(dim)
r  = sqrt(6) / sqrt(dim+1);   % we'll choose weights uniformly from the interval [-r, r]
W = rand(dim, 1) * 2 * r - r;

b = 0;

theta = [W(:); b];