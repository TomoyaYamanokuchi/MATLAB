function v = func_logdet(K, sf)

    % set sf = false
    if nargin == 1
        sf = false;
    end
    
    % decision symmetric matrix
    if issymmetric(K) == 0
        msg = 'K is not a symmetric matrix!';
        error(msg)
    end
    
    % decision positive definite
    L = chol(K);
    
    % calculate log determinant
    % log(det(K)) = 2*sum(log(L))
    v = 2*sum(log(diag(L)));
    
    % decision display
    if sf == true
        calc_result = ['logdet = ', num2str(v)];
        disp(calc_result)
    end
            
end