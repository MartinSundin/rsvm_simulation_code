%Algorithm from "Sparse Bayesian Methods for Low rank matrix estimation", by Babacan,
%Luessi, Molina and Katsaggelos
%Martin Sundin, 2014-02-20

function Xhat = vb_completion(y,AA,p,q,r_prior)

[m,~] = size(AA);
r = r_prior;
S = AA'*ones(m,1);
S = reshape(S,p,q);
Y = AA'*y;
Y = reshape(Y,p,q);
[u,s,v] = svds(Y,r);
A = u*sqrt(s);
B = v*sqrt(s);
Xhat = A*B';
beta = 1;
gamma = eye(r,r);
SigmaA = zeros(r,r,p);
SigmaB = zeros(r,r,q);
for i = 1:p
    SigmaA(:,:,i) = eye(r,r);
end
for i = 1:q
    SigmaB(:,:,i) = eye(r,r);
end

reg1 = 1e-4;
maxiter = 100;
iter = 0;
tol1 = 0.005;
diff = 1;
while (iter < maxiter) && (diff > tol1)
    iter = iter + 1;
    Xold = A*B';
    %Update A
    for i = 1:r
        J = find(S(i,:));
        K = diag(S(i,:));
        Sigma = beta*B'*K*B;
        for j = 1:length(J)
            j2 = J(j);
            Sigma = Sigma + beta*SigmaB(:,:,j2);
        end
        SigmaA(:,:,i) = pinv(Sigma + gamma);
        yi = Y(i,:)';
        a = beta*SigmaA(:,:,i)*B'*yi;
        A(i,:) = a';
    end
    %Update B
    for i = 1:r
        J = find(S(:,i));
        K = diag(S(:,i));
        Sigma = beta*A'*K*A + gamma;
        for j = 1:length(J)
            j2 = J(j);
            Sigma = Sigma + beta*SigmaA(:,:,j2);
        end
        SigmaB(:,:,i) = pinv(Sigma);
        yi = Y(:,i);
        b = beta*SigmaB(:,:,i)*A'*yi;
        B(i,:) = b';
    end
    %Update gamma
    for i = 1:r
        aa = A(:,i)'*A(:,i);
        bb = B(:,i)'*B(:,i);
        for j = 1:r
            aa = aa + SigmaA(i,i,j);
            bb = bb + SigmaB(i,i,j);
        end
        gamma(i,i) = (p+q)/(aa + bb + reg1);
    end
    %Update beta
    Xhat = A*B';
    %Compute diagonal of C matrix
    C = zeros(p*q,p*q);
    for i = 1:p
        for k = 1:q
            C(i+q*(j-1),i+q*(j-1)) = C(i+q*(j-1),i+q*(j-1)) + A(i,:)*SigmaB(:,:,k)*A(i,:)' + B(k,:)*SigmaA(:,:,i)*B(k,:)' + trace(SigmaA(:,:,i)*SigmaB(:,:,k));
            %C + kron(B(:,i)*B(:,i)' + SigmaB(:,:,i),A(:,i)*A(:,i)' + SigmaA(:,:,i)) - kron(B(:,i)*B(:,i)', A(:,i)*A(:,i)');
        end
    end
    beta = m/(norm(y - AA*Xhat(:),2)^2 + trace(AA*C*AA') + reg1);%This is incorrect?!?!
    
    diff = norm(Xold - Xhat,'fro')/norm(Xold,'fro');
end
