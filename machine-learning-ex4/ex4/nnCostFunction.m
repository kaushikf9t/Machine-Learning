function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%size(y) = 5000 by 1
%size(X) = 5000 by 400
%Theta1 = 25*401
% a1 = 5000*26
%Theta2 = 10*26
%hthetaX = 5000*10 = m*k
% 5000*10

X = [ones(m,1), X];
a1 = X;
a2 = sigmoid(X*Theta1');
a2 = [ones(m,1),a2];
hthetaX = sigmoid(a2*Theta2');
a3 = hthetaX;

YMatrix = zeros(m, num_labels);
for i=1:m,
  temp = y(i);
  YMatrix(i, y(i)) = 1;
end

%val1 = -YMatrix(i,:) * log(hthetaX(i,:))';
%val2 = (1-YMatrix(i,:)) * (log(1-hthetaX(i,:)))';

%J += 1/m*(val1 - val2);
val1 = sum(sum(-YMatrix .* log(hthetaX)));
val2 = sum(sum((1-YMatrix).*log(1-hthetaX)));
J = 1/m *(val1 - val2);

Theta1Squared = 0;
Theta2Squared = 0;
for j = 1:hidden_layer_size,
  for k = 2:input_layer_size +1,
    Theta1Squared += (Theta1(j,k) ^2);
  end
end

for j = 1:num_labels,
  for k = 2:hidden_layer_size+1,
    Theta2Squared += (Theta2(j,k) ^2);
  end
end

J+= lambda/(2*m) * (Theta1Squared + Theta2Squared);


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time

Theta1(:,1) = 0;
Theta2(:,1) = 0;
d3 = zeros(m, num_labels);
d2 = zeros(m, hidden_layer_size);
sigGradA2 = zeros(m, hidden_layer_size +1);


%for t = 1:m,
%  d3(t,:) = a3(t,:) - YMatrix(t,:);
%  sigGradA2(t) = a2(t) .*(1-a2(t));
%  intermediateValue = d3(t,:) * Theta2(:,2:hidden_layer_size+1);
%  d2(t,:) = intermediateValue .*sigGradA2(t);

%end

d3 = a3 - YMatrix;
sigGradA2 = a2 .*(1-a2);
intermediateValue = d3 * Theta2(:, 2:hidden_layer_size+1);
d2 = intermediateValue .*sigGradA2(:,2:hidden_layer_size+1);

delta1 = d2' * a1;
delta2 = d3' * a2;

Theta1_grad = delta1/m + (lambda/m * Theta1);
Theta2_grad = delta2/m + (lambda/m * Theta2);


%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
