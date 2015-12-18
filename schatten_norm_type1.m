%Schatten norm type-I estimator
%Code for gradient and objective function
%used as input to for Matlab's fminunc(...)
%Martin Sundin, 2015-08-27

function [J,grad] = schatten_norm_type1(X,A,y,p,q,s,lambda)

X = reshape(X,p,q);
[u,sigma,v] = svd(X);
sigma = diag(sigma);
reg1 = 1e-2;

J = lambda*norm(y - A*X(:),2)^2 + sum(sigma.^s);

grad = 2*lambda*A'*(y - A*X(:));
for k = 1:min(p,q)
    grad = grad + s*abs(sigma(k) + reg1)^(s-1)*kron(v(:,k),u(:,k));
end

%Implement fminunc as
%options = optimset('GradObj', 'on', 'MaxIter', 100);
%Xhat = fminunc(@(t)(schatten_norm_type1(t,A,y,p,q,s,lambda)),pinv(A)*y,options);
%Xhat = reshape(Xhat,p,q);
