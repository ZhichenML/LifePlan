function label = votes(neighbors, train_label)


neighbor_votes = train_label(neighbors);
labels = unique(neighbor_votes);
label_collection = zeros(size(labels));
for i = 1: size(labels)
    label_collection(i) = sum(find(neighbor_votes == labels(i)));
end

[~, index] = max(label_collection);
label = labels(index);