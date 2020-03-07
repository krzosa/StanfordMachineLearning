function [J, grad] = costFunction(theta, X, y)

% number of training examples
m = length(y); 

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

J = (1/m)*((-y'*log(sigmoid(X*theta)))-((1-y)'*log(1-sigmoid(X*theta))));
grad = (1/m)*(sigmoid(X*theta)-y)'*X;

end