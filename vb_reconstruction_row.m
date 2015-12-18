%Variational Bayesian for matrix reconstruction
%A further development of the method from the paper
%"Sparse Bayesian methods for Low-rank matrix estimation", by Babacan,
%Luessi, Molina and Katsaggelos
%Row-wise block partition of factor matrices
%Martin Sundin, 2015-08-26

function [Xhat,Ahat,Bhat] = vb_reconstruction_row(y,Phi,p,q,rmax)

%Initialize variables
[m,~] = size(Phi);
Xhat = reshape(pinv(Phi)*y,p,q);
[u,s,v] = svds(Xhat,rmax);
Ahat = u*sqrt(s);
Bhat = v*sqrt(s);
SigmaA = zeros(rmax,rmax,p);
SigmaB = zeros(rmax,rmax,q);
gamma = ones(rmax,1);
beta = 1;
for k = 1:p
    SigmaA(:,:,k) = diag(gamma);%1./gamma
end
for k = 1:q
    SigmaB(:,:,k) = diag(gamma);%1./gamma
end

reg1 = 1e-5;
maxiter = 5;
min_diff = 1e-5;
iter = 0;
diff = 1;
while (iter < maxiter) && (diff > min_diff)
    iter = iter + 1;
    Xhat_old = Ahat*Bhat';
    %Estimate row vectors in A
    for k = 1:rmax
        %Compute SigmaA and residue
        res = zeros(rmax,1);
        B2k = zeros(rmax,rmax);
        for k2 = 1:m
            res = res + Bhat'*Phi(k2,k:p:end)'*y(k2);
            for j = setdiff(1:p,k)
                res = res - Bhat'*Phi(k2,k:p:end)'*Phi(k2,j:p:end)*Bhat*Ahat(j,:)';
                for k3 = 1:q
                    res = res - Phi(k2,k+(k3-1)*p)*Phi(k2,j+(k3-1)*p)*SigmaB(:,:,k3)*Ahat(j,:)';
                end
            end
            B2k = B2k + Bhat'*Phi(k2,k:p:end)'*Phi(k2,k:p:end)*Bhat;
            for k3 = 1:q
                B2k = B2k + Phi(k2,k+(k3-1)*p)*Phi(k2,k+(k3-1)*p)*SigmaB(:,:,k3);
            end
        end
        SigmaA(:,:,k) = pinv(B2k + diag(gamma));
        ahat = beta*SigmaA(:,:,k)*res;
        Ahat(k,:) = ahat';
    end
    %Estimate row vectors in B
    for k = 1:rmax
        %Compute SigmaA and residue
        res = zeros(rmax,1);
        A2k = zeros(rmax,rmax);
        for k2 = 1:m
            res = res + Ahat'*Phi(k2,1+(k-1)*p:k*p)'*y(k2);
            for j = setdiff(1:q,k)
                res = res - Ahat'*Phi(k2,1+(k-1)*p:k*p)'*Phi(k2,1+(j-1)*p:j*p)*Ahat*Bhat(j,:)';
                for k3 = 1:p
                    res = res - Phi(k2,k3+(k-1)*p)*Phi(k2,k3+(j-1)*p)*SigmaB(:,:,k3)*Bhat(j,:)';
                end
            end
            A2k = A2k + Ahat'*Phi(k2,1+(k-1)*p:k*p)'*Phi(k2,1+(k-1)*p:k*p)*Ahat;
            for k3 = 1:p
                A2k = A2k + Phi(k2,k3+(k-1)*p)*Phi(k2,k3+(k-1)*p)*SigmaB(:,:,k3);
            end
        end
        SigmaB(:,:,k) = pinv(A2k + diag(gamma));
        bhat = beta*SigmaB(:,:,k)*res;
        Bhat(k,:) = bhat';
    end
    
    %Update Gamma parameters
    for k = 1:rmax
        gamma(k) = (p+q+2*reg1)/(norm(Ahat(:,k),2)^2 + norm(Bhat(:,k),2)^2 + sum(SigmaA(k,k,:)) + sum(SigmaB(k,k,:)) + 2*reg1);
    end
    %Update beta
    beta1 = 0;
    for k = 1:m
        for i = 1:p
            beta1 = beta1 + Phi(k,i:p:end)*Bhat*SigmaA(:,:,i)*Bhat'*Phi(k,i:p:end)';
            for j = 1:q
                beta1 = beta1 + Phi(k,i+(j-1)*p)*Phi(k,i+(j-1)*p)*trace(SigmaA(:,:,i)*SigmaB(:,:,j));
            end
        end
        for j = 1:q
            beta1 = beta1 + Phi(k,1+(j-1)*p:j*p)*Ahat*SigmaB(:,:,i)*Ahat'*Phi(k,1+(j-1)*p:j*p)';
        end
    end
    Xhat = Ahat*Bhat';
    beta = (m + 2*reg1)/(norm(y - Phi*Xhat(:),2)^2 + beta1 + 2*reg1);
    diff = norm(Xhat - Xhat_old,'fro')/norm(Xhat,'fro');
end

Xhat = Ahat*Bhat';
