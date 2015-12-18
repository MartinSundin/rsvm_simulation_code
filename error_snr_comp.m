%Plot of NMSE vs. SNR for Matrix completion
%Martin Sundin, 2014-11-11
tic;

%Data is saved in filename.mat
filename = 'snr_comp1_test1';

p = 15;%Height of X
q = 30;%Width of X
r = 3;%Rank of X
s = 0.5;%s value used by RSVM-SN

alpha = 0.7;%alpha = m/pq
M = 10;%Number of sensing matrix realizations
maxiter = 10;%Number of X matrix realizations
SNR_list = 0:5:40%SNR in dB
la = length(SNR_list);
m = round(p*q*alpha);%Number of measurements

%Lists with errors
mse_vb = zeros(la,1);
mse_rvmlog = zeros(la,1);
mse_rvmschatten = zeros(la,1);
mse_nuclear = zeros(la,1);
mse_schatten1 = zeros(la,1);
mse_variational = zeros(la,1);
mse_bayesian_pca = zeros(la,1);
mse_pmf = zeros(la,1);
mse_wtn = zeros(la,1);
x_norms = zeros(la,1);

parfor i = 1:la
    SNR = SNR_list(i);
    disp(['SNR = ' num2str(SNR)]);
    sigma2n = r*10^(-SNR/10);
    sigman = sqrt(sigma2n);
    for m1 = 1:M
        %Generate sensing matrix
        A = zeros(m,p*q);
        J = randperm(p*q);
        J = sort(J(1:m));
        A(:,J) = eye(m,m);
        for iter = 1:maxiter
            %Generate low-rank matrix X
            L = randn(p,r);
            R = randn(r,q);
            X = L*R;
            x_norms(i) = x_norms(i) + norm(X,'fro')^2;
            
            %Generate measurements
            y = A*X(:) + sigman*randn(m,1);
            Y = zeros(size(X));
            Y(J) = y;
            
            %VB
            Xhat = vb_completion(y,A,p,q,r);
            mse_vb(i) = mse_vb(i) + norm(Xhat - X,'fro')^2;
            
            %Schatten
            Xhat = rsvm_schatten(y,A,p,q,s);
            mse_rvmschatten(i) = mse_rvmschatten(i) + norm(Xhat - X,'fro')^2;
            
            %RSVM
            Xhat = rsvm_ld(y,A,p,q);
            mse_rvmlog(i) = mse_rvmlog(i) + norm(Xhat - X,'fro')^2;
            
            %Nuclear norm
            lambda = sigman*sqrt(m+sqrt(8*m));
            Xhat = nuclear_norm(y,A,p,q,lambda);
            mse_nuclear(i) = mse_nuclear(i) + norm(Xhat - X,'fro')^2;
            
            %Type-I Schatten norm
            try
                options = optimset('GradObj', 'on', 'MaxIter', 100,'Display', 'off');
                Xhat = fminunc(@(t)(schatten_norm_type1(t,A,y,p,q,s,lambda)),pinv(A)*y,options);
                Xhat = reshape(Xhat,p,q);
                mse_schatten1(i) = mse_schatten1(i) + norm(Xhat - X,'fro')^2;
            catch ME
                disp('Type-I Schatten error');
                disp(ME);
                mse_schatten1(i) = mse_schatten1(i) + norm(X,'fro')^2;
            end

            %Variational Movie Rating
            Xhat = variational_movierating(Y);
            mse_variational(i) = mse_variational(i) + norm(Xhat - X,'fro')^2;

            %Bayesian PCA
            Xhat = bayesian_pca(Y);
            mse_bayesian_pca(i) = mse_bayesian_pca(i) + norm(Xhat - X,'fro')^2;

            %Probabilistic Matrix Factorization
            Xhat = prob_matrix_fact(Y);
            mse_pmf(i) = mse_pmf(i) + norm(Xhat - X,'fro')^2;

            %Weighted Trace norm
            Xhat = weighted_trace_norm(Y,lambda);
            mse_wtn(i) = mse_wtn(i) + norm(Xhat - X,'fro')^2;
        end
    end
end

%Compute NMSE
nmse_vb = mse_vb./x_norms;
nmse_rvmschatten = mse_rvmschatten./x_norms;
nmse_rvmlog = mse_rvmlog./x_norms;
nmse_nuclear = mse_nuclear./x_norms;
nmse_schatten1 = mse_schatten1./x_norms;
nmse_variational = mse_variational./x_norms;
nmse_bayesian_pca = mse_bayesian_pca./x_norms;
nmse_pmf = mse_pmf./x_norms;
nmse_wtn = mse_wtn./x_norms;

%Save results
save([filename '.mat'],'p','q','r','s','SNR_list','M','maxiter','nmse_vb','nmse_rvmschatten','nmse_rvmlog','nmse_nuclear','x_norms','nmse_schatten1','nmse_variational','nmse_bayesian_pca','nmse_pmf','nmse_wtn');

%Plot results
figure;
hold on;
plot(SNR_list,10*log10(nmse_vb),'-ob','linewidth',2);
plot(SNR_list,10*log10(nmse_rvmschatten),'-sk','linewidth',2);
plot(SNR_list,10*log10(nmse_rvmlog),'-sb','linewidth',2);
plot(SNR_list,10*log10(nmse_nuclear),'-dc','linewidth',2);
plot(SNR_list,10*log10(nmse_schatten1),'-<g');
plot(SNR_list,10*log10(nmse_variational),'-*g');
plot(SNR_list,10*log10(nmse_pmf),'-sr');
plot(SNR_list,10*log10(nmse_wtn),'-or');
legend('VB-1','RSVM-SN','RSVM-LD','Nuclear norm','SNA','VB-2','PMF','WTN','Location','Best');
ylabel('NMSE [dB]');
xlabel('SNR [dB]');
box on;

myfontname = 'Arial';
set(gca,'FontSize',9,'fontName',myfontname);
set(findall(gcf,'type','text'),'FontSize',9,'fontName',myfontname);
lineobj = findobj('type', 'line');
set(lineobj, 'linewidth', 1.8);

toc;
