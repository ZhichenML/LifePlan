function neighbors = get_neighbors(training, newobservation,k)

N = size(training,1);
newobservation = repmat(newobservation,N,1);
dists = (training - newobservation).^2;
dists = sum (dists,2);
[~, neighbors] = sort(dists);
disp(k)
neighbors = neighbors(1:k);

end