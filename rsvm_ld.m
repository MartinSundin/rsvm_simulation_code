%RSVM-LD
%Martin Sundin, 2014-06-12

function Xhat = rsvm_ld(y,A,p,q)

[m,~] = size(A);
beta = 1;
alphaL = eye(p,p);
alphaR = eye(q,q);
ialphaL = alphaL;
ialphaR = alphaR;
Xhat = zeros(p,q);
AtA = sparse(A)'*sparse(A);
Aty = A'*y;

reg1 = 1e-3;
maxiter = 25;
mindiff = 1e-2;
iter = 0;
diff = 1;
while (iter < maxiter) && (diff > mindiff)
    iter = iter + 1;
    Xold = Xhat;
    try
        alphaL = pinv(ialphaL);
        alphaR = pinv(ialphaR);
        %Compute Sigma matrix
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

        ialphaL2 = SigmaR + Xhat*alphaR*Xhat' + 1*reg1*eye(p,p);
        ialphaR2 = SigmaL + Xhat'*alphaL*Xhat + 1*reg1*eye(q,q);
    
        %Balance precisions
        tL = trace(ialphaL2);
        tR = trace(ialphaR2);
        h = sqrt(tL/tR);
        g = sqrt((tL*tR)/(norm(Xhat,'fro')^2 + trace(Sigma)));
    
        ialphaL = ialphaL2/(h*g);
        ialphaR = ialphaR2*h/g;
    catch ME
        disp(['Aborted at iteration ' num2str(iter)])
        iter = maxiter;
        disp(['Error: ' ME.message]);
    end
    
    if sum(sum(isinf(alphaL))) + sum(sum(isinf(alphaR))) + sum(sum(isinf(beta))) + sum(sum(isnan(alphaL))) + sum(sum(isnan(alphaR))) + sum(sum(isnan(beta))) > 0
        iter = maxiter;
        disp('Nan or Inf found');
        Xhat = Xold;
    end
    
    diff = norm(Xold - Xhat,'fro')/norm(Xold,'fro');
end
