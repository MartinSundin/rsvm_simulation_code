%Probabilistic Matrix Factorization
%From paper "Probabilistic Matrix Factorization", by R. Salakhutdinov and
%A. Mnih
%Uses logistic map to find to estimate missing entries
%Martin Sundin, 2015-09-11

function Xhat = prob_matrix_fact(Y)

[N,M] = size(Y);
%K = nnz(Y);
D = min(N,M);
I = (Y ~= 0);

%Scale data
Ys = (Y - min(min(Y)))/(max(max(Y)) - min(min(Y)));

%Initialize method
[u,s,v] = svds(Y,D);
U = sqrt(s)*u';
V = sqrt(s)*v';

%Regularization parameters
lambdaU = 0.002;
lambdaV = 0.002;

iter = 0;
maxiter = 100;
mindiff = 1e-3;
diff1 = 1;
while (iter < maxiter) && (diff1 > mindiff)
    iter = iter + 1;
    Xhat_old = 1./(1 + exp(-U'*V));
    g = Xhat_old;
    dU = - I.*(Ys - g).*g.*(1-g);
    dV = U*dU + lambdaV*V;
    dU = dU*V' + lambdaU*U;
    steplength = 0.1;
    if iter > 10
        steplength = 1/iter;
    end
    U = U - steplength*dU;
    V = V - steplength*dV;
    Xs = 1./(1 + exp(-U'*V));
    diff1 = norm(Xhat_old - Xs,'fro');
end

%Rescale data
Xhat = min(min(Y)) + (max(max(Y)) - min(min(Y)))*Xs;
