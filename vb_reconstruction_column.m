%Variational Bayesian for matrix reconstruction
%A further development of the method from the paper
%"Sparse Bayesian methods for Low-rank matrix estimation", by Babacan,
%Luessi, Molina and Katsaggelos
%Columnwise block partition of factor matrices
%Martin Sundin, 2015-08-24

function [Xhat,Ahat,Bhat] = vb_reconstruction_column(y,Phi,p,q,rmax)

%Initialize variables
[m,~] = size(Phi);
Xhat = reshape(pinv(Phi)*y,p,q);
[u,s,v] = svds(Xhat,rmax);
Ahat = u*sqrt(s);
Bhat = v*sqrt(s);
SigmaA = zeros(p,p,rmax);
SigmaB = zeros(q,q,rmax);
gamma = ones(rmax,1);%2./(diag(Ahat'*Ahat) + diag(Bhat'*Bhat));%ones(rmax,1);
%Xhat = Ahat*Bhat';
beta = 1;%1/norm(y - Phi*Xhat(:),2)^2;
for k = 1:rmax
    SigmaA(:,:,k) = gamma(k)*eye(p,p);
    SigmaB(:,:,k) = gamma(k)*eye(q,q);
end

reg1 = 1e-5;
maxiter = 5;
min_diff = 1e-5;
iter = 0;
diff = 1;
while (iter < maxiter) && (diff > min_diff)
    iter = iter + 1;
    Xhat_old = Ahat*Bhat';
    Ahat_old = Ahat;
    Bhat_old = Bhat;
    %Estimate vectors in A and B
    for k = 1:rmax
        B2 = Bhat_old(:,k)*Bhat_old(:,k)' + SigmaB(:,:,k);
        A2 = Ahat_old(:,k)*Ahat_old(:,k)' + SigmaA(:,:,k);
        Xk = Ahat_old*Bhat_old' - Ahat_old(:,k)*Bhat_old(:,k)';
        rk = y - Phi*Xk(:);
        bta = zeros(p,1);
        btb = zeros(q,1);
        SigmaAk = zeros(p,p);
        SigmaBk = zeros(q,q);
        for m1 = 1:m
            Phik = reshape(Phi(k,:),p,q);
            SigmaAk = SigmaAk + Phik*B2*Phik';
            SigmaBk = SigmaBk + Phik'*A2*Phik;
            bta = bta + rk(m1)*Phik*Bhat_old(:,k);
            btb = btb + rk(m1)*Phik'*Ahat_old(:,k);
        end
        SigmaAk = beta*SigmaAk + gamma(k)*eye(p,p);
        SigmaBk = beta*SigmaBk + gamma(k)*eye(q,q);
        SigmaA(:,:,k) = pinv(SigmaAk);
        SigmaB(:,:,k) = pinv(SigmaBk);
        Ahat(:,k) = beta*SigmaA(:,:,k)*bta;
        Bhat(:,k) = beta*SigmaB(:,:,k)*btb;
    end
    %Balance matrices
    alpha = sqrt(norm(Bhat,'fro')/norm(Ahat,'fro'));
    Ahat = alpha*Ahat;
    Bhat = Bhat/alpha;
    
    %Update Gamma parameters
    for k = 1:rmax
        gamma(k) = (p+q+2*reg1)/(norm(Ahat(:,k),2)^2 + norm(Bhat(:,k),2)^2 + trace(SigmaA(:,:,k)) + trace(SigmaB(:,:,k)) + 2*reg1);
    end
    %Update beta
    beta1 = 0;
    for k = 1:m
        Phik = reshape(Phi(k,:),p,q);
        for i = 1:rmax
            beta1 = beta1 + Bhat(:,i)'*Phik'*SigmaA(:,:,i)*Phik*Bhat(:,i) + Ahat(:,i)'*Phik*SigmaB(:,:,i)*Phik'*Ahat(:,i) + trace(Phik*SigmaB(:,:,i)*Phik'*SigmaA(:,:,i));
        end
    end
    Xhat = Ahat*Bhat';
    beta = (m + 2*reg1)/(norm(y - Phi*Xhat(:),2)^2 + beta1 + 2*reg1);
    diff = norm(Xhat - Xhat_old,'fro')/norm(Xhat,'fro');
end

Xhat = Ahat*Bhat';
