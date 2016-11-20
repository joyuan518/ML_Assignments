function [C, sigma] = dataset3Params(X, y, Xval, yval)
  %EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
  %where you select the optimal (C, sigma) learning parameters to use for SVM
  %with RBF kernel
  %   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
  %   sigma. You should complete this function to return the optimal C and 
  %   sigma based on a cross-validation set.
  %

  % You need to return the following variables correctly.
  C = 1;
  sigma = 0.3;

  % ====================== YOUR CODE HERE ======================
  % Instructions: Fill in this function to return the optimal C and sigma
  %               learning parameters found using the cross validation set.
  %               You can use svmPredict to predict the labels on the cross
  %               validation set. For example, 
  %                   predictions = svmPredict(model, Xval);
  %               will return the predictions on the cross validation set.
  %
  %  Note: You can compute the prediction error using 
  %        mean(double(predictions ~= yval))
  %

  params = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];  
  params_length = length(params);
  error_min = -1;
  zero_found = false;
  
  for i = 1:params_length
      for j = 1:params_length
        C_current = params(i);
        sigma_current = params(j);
        
        model= svmTrain(X, y, C_current, @(x1, x2) gaussianKernel(x1, x2, sigma_current));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
                
        if error_min == -1 | error < error_min
           error_min = error;        
           C = C_current;
           sigma = sigma_current;
           
           % zero is the minimum error possible, so don't need to continue
           if error == 0
              zero_found = true;
              break;
           end           
        end        
      end
      
      if (zero_found)
         break;
      end
  end
  
  % =========================================================================

end
