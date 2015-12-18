%RSVM-SN for two-sided precision
%Martin Sundin, 2014-06-12

function Xhat = rsvm_schatten(y,A,p,q,s)

[m,~] = size(A);
beta = 1;
ialphaL = eye(p,p);
ialphaR = eye(q,q);
Xhat = zeros(p,q);
AtA = A'*A;
Aty = A'*y;

reg1 = 1e-3;
reg2 = 1e-5;
maxiter = 25;
mindiff = 1e-2;
iter = 0;
diff = 1;
while (iter < maxiter) && (diff > mindiff)
    iter = iter + 1;
    Xold = Xhat;
    
    try
        alphaR = pinv(ialphaR);
        alphaL = pinv(ialphaL);
        Sigma = pinv(kron(alphaR,alphaL) + beta*AtA);
        
        %Update estimate
        Xhat = beta*(Sigma*Aty);
        Xhat = reshape(Xhat,p,q);
        
        %Update precisions
        beta = (m + 2*reg1)/(norm(y-A*Xhat(:),2)^2 + sum(sum(Sigma.*AtA)) + 2*reg1);

        %Form SigmaR and SigmaL
        SigmaR = zeros(p,p);
        SigmaL = zeros(q,q);

        S = Sigma*kron(speye(q,q),alphaL);
        for k = 1:p
            SigmaL = SigmaL + S(k:p:end,k:p:end);
        end

        S = Sigma*kron(alphaR,speye(p,p));
        for k = 1:q
            SigmaR = SigmaR + S(1+(k-1)*p:k*p,1+(k-1)*p:k*p);
        end

        [U,S,~] = svd(SigmaR + Xhat*alphaR*Xhat');
        S = diag(S);
        ialphaL2 = U*diag((S+reg2).^(-(s-2)/2))*U';
        tL = sum((S+reg2).^(-(s-2)/2)) + reg2;
    
        [U,S,~] = svd(SigmaL + Xhat'*alphaL*Xhat);
        S = diag(S);
        ialphaR2 = U*diag((S+reg2).^(-(s-2)/2))*U';
        tR = sum((S+reg2).^(-(s-2)/2)) + reg2;
        
        %Balance precisions
        h = sqrt(tL/tR);
        g = sqrt((tL*tR)/(norm(Xhat,'fro')^2 + trace(Sigma)));
        ialphaL = ialphaL2/(h*g);
        ialphaR = ialphaR2*h/g;
    catch ME
        disp(['Aborted at iteration ' num2str(iter)])
        iter = maxiter;
        disp(['Error: ' ME.message]);
    end
    
    if sum(sum(isinf(ialphaL))) + sum(sum(isinf(ialphaR))) + sum(sum(isinf(beta))) + sum(sum(isnan(ialphaL))) + sum(sum(isnan(ialphaR))) + sum(sum(isnan(beta))) > 0
        iter = maxiter;
        disp('Nan of Inf found');
        Xhat = Xold;
    end
    
    diff = norm(Xold - Xhat,'fro')/norm(Xold,'fro');
end
