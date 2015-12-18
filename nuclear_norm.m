%Nuclear norm minimization
%uses the cvx toolbox for Matlab
%Martin Sundin, 2014-01-31

function Xhat = nuclear_norm(y,A,p,q,lambda)

cvx_begin sdp quiet
    variable X(p,q);
    minimize norm_nuc(X);
    subject to
        norm(y - A*X(:),2) <= lambda;
cvx_end

Xhat = X;
