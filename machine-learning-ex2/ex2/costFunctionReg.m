function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));
size(grad)

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

hthetaX = sigmoid(X*theta);


theta(1) = 0;
costReg = lambda/(2*m) * sum(realpow(theta, 2))
J = 1/m* (-y'*log(hthetaX) - (1-y)' * log(1-hthetaX)) + costReg;

gradReg = (lambda/m)*theta;

grad = 1/m*(X' * (hthetaX - y)) + gradReg;


% =============================================================

end
