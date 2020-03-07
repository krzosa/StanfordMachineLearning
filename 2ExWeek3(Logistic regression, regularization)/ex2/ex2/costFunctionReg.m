function [J, grad] = costFunctionReg(theta, X, y, lambda)
    m = length(y); % number of training examples
    
    J = 0;
    grad = zeros(size(theta));
    thetaZero = theta;
    thetaZero(1) = 0;
    h_theta = sigmoid(X*theta);
    
    % COST
    J = (1/m) * ((-y'*log(h_theta))-(1-y)'*(log(1-h_theta))) + ((lambda/(2*m))*theta(2:end)'*theta(2:end));
    % GRADIENT
    grad = ((1/m)*X'*(h_theta-y)) + ((lambda/m)*thetaZero);
end