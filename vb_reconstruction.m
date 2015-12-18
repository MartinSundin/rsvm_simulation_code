%Variational Bayesian for matrix reconstruction
%A further development of the method from the paper
%"Sparse Bayesian methods for Low-rank matrix estimation", by Babacan,
%Luessi, Molina and Katsaggelos
%No matrix partition
%Martin Sundin, 2015-08-24

function [Xhat,Ahat,Bhat] = vb_reconstruction(y,Phi,p,q,rmax)

%Initialize variables
[m,~] = size(Phi);
Xhat = reshape(pinv(Phi)*y,p,q);
[u,s,v] = svds(Xhat,rmax);
Ahat = u*sqrt(s);
Bhat = v*sqrt(s);
SigmaA = eye(p*rmax,p*rmax);
SigmaB = eye(q*rmax,q*rmax);
gamma = ones(rmax,1);
beta = 1;
T = transpose_operator(q,rmax);
Py = Phi'*y;
PtP = Phi'*Phi;

reg1 = 1e-5;
maxiter = 5;
min_diff = 1e-2;
iter = 0;
diff = 1;
while (iter < maxiter) && (diff > min_diff)
    iter = iter + 1;
    Xhat_old = Ahat*Bhat';
    Ahat_old = Ahat;
    Bhat_old = Bhat;
    try
    %Update A
    SigmaA = zeros(p*rmax,p*rmax);
    for k = 1:m
        SigmaA = SigmaA + kron(speye(rmax,rmax),reshape(Phi(k,:),p,q))*SigmaB*kron(speye(rmax,rmax),reshape(Phi(k,:),p,q)');
    end
    %keyboard
    SigmaA = pinv(full(beta*SigmaA + beta*kron(Bhat',speye(p,p))*PtP*kron(Bhat,speye(p,p)) + kron(speye(p,p),diag(gamma))));
    Ahat = beta*SigmaA*kron(Bhat',speye(p,p))*Py;
    Ahat = full(reshape(Ahat,p,rmax));
    %Update B
    SigmaB = zeros(q*rmax,q*rmax);
    for k = 1:m
        SigmaB = SigmaB + kron(speye(rmax,rmax),reshape(Phi(k,:),p,q)')*SigmaA*kron(speye(rmax,rmax),reshape(Phi(k,:),p,q));
    end
    SigmaB = pinv(full(beta*SigmaB + beta*T*kron(speye(q,q),Ahat')*PtP*kron(speye(q,q),Ahat)*T + kron(speye(q,q),diag(gamma))));
    Bhat = beta*SigmaB*T*kron(speye(q,q),Ahat')*Py;
    Bhat = full(reshape(Bhat,q,rmax));
    
    %Update Gamma parameters
    for k = 1:rmax
        gamma(k) = (p+q+2*reg1)/(norm(Ahat(:,k),2)^2 + norm(Bhat(:,k),2)^2 + trace(SigmaA(1+(k-1)*p:k*p,1+(k-1)*p:k*p)) + trace(SigmaB(1+(k-1)*q:k*q,1+(k-1)*q:k*q)) + 2*reg1);
    end
    %Update beta
    beta = 0;
    for k = 1:m
        for i = 1:rmax
            beta = beta + trace(SigmaA*kron(speye(rmax,rmax),reshape(Phi(k,:),p,q))*SigmaB*kron(speye(rmax,rmax),reshape(Phi(k,:),p,q)'));
        end
    end
    Xhat = Ahat*Bhat';
    beta = (m + 2*reg1)/(norm(y - Phi*Xhat(:),2)^2 + beta + 2*reg1);
    diff = norm(Xhat - Xhat_old,'fro')/norm(Xhat,'fro');
    catch
        Ahat = Ahat_old;
        Bhat = Bhat_old;
        disp('VB: Error detected.');
        iter = maxiter;
    end 
end

Xhat = Ahat*Bhat';
