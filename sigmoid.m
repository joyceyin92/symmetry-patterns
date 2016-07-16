function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).
[M,N]=size(z);
for m=1:M
    for n=1:N
        g(m,n) = 1/(1+exp(-z(m,n)));
    end
end




% =============================================================

end
