%Bayesian PCA method for matrix completion
%Y is a matrix with 0 for missing values
%From paper "Principal Component nalysis for Large Scale Problems with Lots
%of Missing Values", by T. Raiko, A. Ilin and J. Karhunen
%Martin Sundin, 2015-09-11

function Xhat = bayesian_pca(Y)

[d,n] = size(Y);
c = min(d,n);
K = length(find(Y));

[u,s,v] = svds(Y,c);
A = u*sqrt(s);
S = sqrt(s)*v';
Atilde = ones(d,c);
Stilde = ones(c,n);
vx = 1;
vsk = ones(c,1);

maxiter = 1000;
iter = 0;
mindiff = 1e-3;
diff1 = 1;
while (iter < maxiter) && (diff1 > mindiff)
    iter = iter + 1;
    Xhat_old = A*S;
    %Compute Atilde and Stilde
    Atilde2 = zeros(d,c);
    Stilde2 = zeros(c,n);
    for i = 1:d
        for k = 1:c
            for j = 1:n
                if Y(i,j) ~= 0
                    Atilde2(i,k) = Atilde2(i,k) + (S(k,j).^2 + Stilde(k,j))/vx;
                    Stilde2(k,j) = Stilde2(k,j) + (A(i,k).^2 + Atilde(i,k))/vx;
                end
            end
        end
    end
    Atilde = 1./(1 + Atilde2);
    Stilde = 1./(1 + Stilde2);
    %Compute gradients
    dA = zeros(d,c);
    dS = zeros(c,n);
    R = Y - A*S;
    for l = 1:c
        for i = 1:d
            for j = 1:n
                if Y(i,j) ~= 0
                    dA(i,l) = dA(i,l) + (-R(i,j)*S(l,j) + A(i,l)*Stilde(l,j))/vx;
                    dS(l,j) = dS(l,j) + (-R(i,j)*A(i,l) + Atilde(i,l)*S(l,j))/vx;
                end
            end
        end
    end
    dA = dA + A;
    dS = dS + diag(1./vsk)*S;
    if iter < 100
        steplength = 0.1;
    else
        steplength = 1/iter;
    end
    %Update variances
    vsk = sum(S.^2 + Stilde,2)/n;
    vx = 0;
    for i = 1:d
        for j = 1:n
            if Y(i,j) ~= 0
                vx = vx + R(i,j)^2;
                for k = 1:c
                    vx = vx + Atilde(i,k)*S(k,j)^2 + A(i,k)^2*Stilde(k,j) + Atilde(i,k)*Stilde(k,j);
                end
            end
        end
    end
    vx = vx/K;
    
    %Perform gradient steps
    A = A - steplength*dA;
    S = S - steplength*dS;
    diff1 = norm(Xhat_old - A*S,'fro');
end

Xhat = A*S;