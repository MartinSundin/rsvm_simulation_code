%Variational Bayesian Approach to Movie Rating Prediction
%Algorithm from the paper "Variational Bayesian Approach to Movie Rating
%Prediction", by Yew Jin Lim and Yee Whye Teh, KDDCup 2007
%Martin Sundin, 2015-09-10

function Xhat = variational_movierating(Y)
%Y is the matrix to be completed where Y(i,j) = 0 denotes missing entries

[I,J] = size(Y);
Kl = find(Y);
K = length(Kl);
n = min(I,J);%Upper bound on rank
tau2 = 1;
sigma2 = ones(n,1);
Psi = zeros(n,n,J);
Phi = zeros(n,n,I);
for k = 1:J
    Psi(:,:,k) = eye(n,n);
end
for k = 1:I
    Phi(:,:,k) = eye(n,n);
end
U = randn(I,n);
V = randn(J,n);

iter = 0;
maxiter = 25;
diff1 = 1;
mindiff = 1e-2;
while (iter < maxiter) && (diff1 > mindiff)
    iter = iter + 1;
    Xhat_old = U*V';
    try
        %Update U and V
        for j = 1:J
            S = zeros(n,n,J);
            for j2 = 1:J
                S(:,:,j2) = n*eye(n,n);
            end
            t = zeros(J,n);
            for i = 1:I
                Phii = diag(1./sigma2);
                uhat = zeros(n,1);
                for j2 = 1:J
                    if Y(i,j2) ~= 0
                        Phii = Phii + (Psi(:,:,j2) + V(j2,:)'*V(j2,:))/tau2;
                        uhat = uhat + Y(i,j2)*V(j2,:)'/tau2;
                    end
                end
                Phii = pinv(Phii);
                Phi(:,:,i) = Phii;
                U(i,:) = uhat'*Phii;
                for j2 = 1:J
                    if Y(i,j2) ~= 0
                        S(:,:,j2) = S(:,:,j2) + (Phii + U(i,:)'*U(i,:))/tau2;
                        t(j2,:) = t(j2,:) + Y(i,j)*U(i,:)/tau2;
                    end
                end
            end
            Psi(:,:,j) = pinv(S(:,:,j));
            V(j,:) = (Psi(:,:,j)*t(j,:)')';
        end
        Xhat = U*V';
        Xhat(Kl) = Y(Kl);
        diff1 = norm(Xhat_old - Xhat,'fro');
        %Update variances
        sigma2 = zeros(n,1);
        tau2 = 0;
        for i = 1:n
            sigma2 = sigma2 + diag(Phi(:,:,i)) + U(i,:)'.^2;
        end
        sigma2 = sigma2/(I-1);
        for i = 1:I
            for j = 1:J
                if Y(i,j) ~= 0
                    tau2 = tau2 + Y(i,j)^2 - 2*Y(i,j)*U(i,:)*V(j,:)' + trace((Phi(:,:,i) + U(i,:)'*U(i,:))*(Psi(:,:,j) + V(j,:)'*V(j,:)));
                end
            end
        end
        tau2 = tau2/(K-1);
    catch
        Xhat = Xhat_old;
        iter = maxiter;
    end
end


