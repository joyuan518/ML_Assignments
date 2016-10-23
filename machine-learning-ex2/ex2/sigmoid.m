function g = sigmoid(z)
  %SIGMOID Compute sigmoid functoon
  %   J = SIGMOID(z) computes the sigmoid of z.

  % You need to return the following variables correctly 
  g = zeros(size(z));

  % ====================== YOUR CODE HERE ======================
  % Instructions: Compute the sigmoid of each value of z (z can be a matrix,
  %               vector or scalar).
  
  rn = size(z)(1); % row number
  cn = size(z)(2); % column number
  
  for i = 1:rn
    for j = 1:cn
      g(i,j) = 1 / (1 + e^-z(i,j));
    end
  end

  % =============================================================

end

  
