function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
    lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64)
% hiddenSize: the number of hidden units (probably 25)
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example.

% The input theta is a vector (because minFunc expects the parameters to be a vector).
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this
% follows the notation convention of the lecture notes.

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values).
% Here, we initialize them to zeros.
cost = 0;
W1grad = zeros(size(W1));
W2grad = zeros(size(W2));
b1grad = zeros(size(b1));
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b)
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
%
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2.
%
    a1 = sigmoid(W1*data+b1(:)*ones(1,size(data,2)));
    a2 = sigmoid(W2*a1+b2(:)*ones(1,size(data,2)));
    % average square
    square_ave = sum(sum((a2-data).*(a2-data)/2))/size(data,2);
    % weight decay
    weight_decay = lambda/2*(sum(sum(W1.^2)) + sum(sum(W2.^2)));
    % sparsity
    ave_sparsity = sum(a1,2)/size(data,2);
    sparsity = beta*(sum(sparsityParam*log(sparsityParam./ave_sparsity)) + ...
        sum((1-sparsityParam)*log((1-sparsityParam)./(1-ave_sparsity))));
    
    cost = square_ave + weight_decay + sparsity;
    
    delta_visible = (a2-data).*a2.*(1-a2);
    % sparsity for each example
    sparsity_target = sparsityParam * ones(hiddenSize,size(data,2));
    sparsity_ave = ave_sparsity * ones(1,size(data,2));
    sparsity_penalty = beta*(-sparsity_target./sparsity_ave+ ((1-sparsity_target)./(1-sparsity_ave)));
    delta_hidden = (W2' * delta_visible + sparsity_penalty ) .* a1 .*(1-a1);
    
    W2grad =  delta_visible*a1'/size(data,2) + lambda .* W2;
    W1grad = delta_hidden*data'/size(data,2) + lambda .* W1;
    b2grad = sum(delta_visible,2)/size(data,2);
    b1grad = sum(delta_hidden,2)/size(data,2);

% W2grad = W2grad/numpatch;
% W1grad = W1grad/numpatch;
% b2grad = b2grad/numpatch;
% b1grad = b1grad/numpatch;


%============================================
% numpatch = numel(data(1,:));
% 
% for i = 1:numpatch
%     a1 = sigmoid(W1*data(:,i)+b1);
%     a2 = sigmoid(W2*a1+b2);
%     cost = (a2-data(:,i))'*(a2-data(:,i))/2;
% %     delta_hidden = nan(hiddenSize,1);
% %     delta_visible = nan(visibleSize,1);
%     delta_visible = (a2-data(:,i)).*a2.*(1-a2);
%     delta_hidden = (W2' * delta_visible) .* a1 .*(1-a1);
%     W2grad = W2grad + delta_visible*a1';
%     W1grad = W1grad + delta_hidden*data(:,i)';
%     b2grad = b2grad + delta_visible;
%     b1grad = b1grad + delta_hidden;
%     % check gradient
%     check = 0;
%     if check
%         grad = [W1grad(:);W2grad(:);b1grad(:);b2grad(:)];
%         eps = 10^(-4);
%         para = [W1(:); W2(:); b1(:); b2(:);];
%         numgrad = zeros(size(para));
%         for ind = 1:numel(numgrad)
%             temp = zeros(numel(numgrad),1); temp(ind) = eps;
%             low = para - temp;
%             up = para + temp;
%             
%             [W1_low,W2_low,b1_low,b2_low]=devector(low,hiddenSize,visibleSize);
%             [W1_up,W2_up,b1_up,b2_up]=devector(up,hiddenSize,visibleSize);
%             
%             a1_low = sigmoid(W1_low*data(:,i)+b1_low);
%             a2_low = sigmoid(W2_low*a1_low+b2_low);
%             val_low = (a2_low-data(:,i))'*(a2_low-data(:,i))/2;
%             
%             a1_up = sigmoid(W1_up*data(:,i)+b1_up);
%             a2_up = sigmoid(W2_up*a1_up+b2_up);
%             val_up = (a2_up-data(:,i))'*(a2_up-data(:,i))/2;
%             
%             numgrad(ind) = (val_up-val_low)/(2*eps);
%             
%             
%       
%         end
%         check = [grad numgrad];
%     end
% end
% W2grad = W2grad/numpatch;
% W1grad = W1grad/numpatch;
% b2grad = b2grad/numpatch;
% b1grad = b1grad/numpatch;














%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)).

function sigm = sigmoid(x)

sigm = 1 ./ (1 + exp(-x));
end

function val = Autoencoder(W1,W2,b1,b2,x)
a1 = sigm(W1*x+b1);
a2 = sigm(W2*a2+b2);
val = sum(a2);
end

function [W1,W2,b1,b2]=devector(theta,hiddenSize,visibleSize)
W1 = reshape(theta(1:hiddenSize*visibleSize),hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize),visibleSize,hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);
end