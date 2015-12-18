%Weighted Trace norn for matrix completion
%From paper "Collaborative Filtering in a Non-Uniform World: Learning with
%the Weighted Trace Norm", by R. Salakhutdinov and N. Srebro
%Uses cvx toolbox to perform minimization
%Martin Sundin, 2015-09-14

function Xhat = weighted_trace_norm(Y,lambda)

[M,N] = size(Y);
J = find(Y);
W = (Y ~= 0);
p = sum(W,2)/N;
q = sum(W,1)/M;

cvx_begin sdp quiet
    variable X(M,N);
    minimize norm_nuc(diag(sqrt(p))*X*diag(sqrt(q)));
    subject to
        norm(Y(J) - X(J),2) <= lambda;
cvx_end

Xhat = X;
