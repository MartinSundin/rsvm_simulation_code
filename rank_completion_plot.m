%Plot of NMSE vs. SNR for Matrix completion
%Martin Sundin, 2014-04-12
tic;

%Data is saved in filename.mat
filename = 'rank_comp1_test1';

p = 15;%Height of X
q = 30;%Width of X
s = 0.5;%s value used by RSVM-SN
alpha = 0.7;%alpha = m/pq
r_list = 1:1:10%Rank of X

M = 10;%Number of sensing matrix realizations
maxiter = 10;%Number of X matrix realizations
SNR = 20;%SNR in dB
la = length(r_list);
m = round(p*q*alpha);%Number of measurements

%Lists with errors
mse_vb = zeros(la,1);
mse_rvmlog2 = zeros(la,1);
mse_rvmschatten = zeros(la,1);
mse_nuclear = zeros(la,1);
mse_schatten1 = zeros(la,1);
mse_variational = zeros(la,1);
mse_bpca = zeros(la,1);
mse_pmf = zeros(la,1);
mse_wtn = zeros(la,1);
x_norms = zeros(la,1);

parfor i = 1:la
    r = r_list(i);
    disp(['rank = ' num2str(r)]);
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
            X = randn(p,r)*randn(r,q);
            x_norms(i) = x_norms(i) + norm(X,'fro')^2;
            
            %Generate measurements
            y = A*X(:) + sigman*randn(m,1);
            Y = zeros(p,q);
            Y(J) = y;
            
            %Variational Bayesian
            Xhat = vb_completion(y,A,p,q,r);%min(p,q));
            mse_vb(i) = mse_vb(i) + norm(Xhat - X,'fro')^2;
            
            %RSVM
            Xhat = rsvm_ld(y,A,p,q);
            mse_rvmlog2(i) = mse_rvmlog2(i) + norm(Xhat - X,'fro')^2;
            
            %Schatten
            Xhat = rvm_schatten(y,A,p,q,s);
            mse_rvmschatten(i) = mse_rvmschatten(i) + norm(Xhat - X,'fro')^2;
            
            %Nuclear norm
            lambda = sigman*sqrt(m+sqrt(8*m));
            Xhat = nuclear_norm(y,A,p,q,lambda);
            mse_nuclear(i) = mse_nuclear(i) + norm(Xhat - X,'fro')^2;
            
            %Type-I Schatten
            options = optimset('GradObj', 'on', 'MaxIter', 100,'Display','off');
            Xhat = fminunc(@(t)(schatten_norm_type1(t,A,y,p,q,s,1)),pinv(A)*y,options);
            Xhat = reshape(Xhat,p,q);
            mse_schatten1(i) = mse_schatten1(i) + norm(Xhat - X,'fro')^2;
            
            %Variational Movie Rating
            Xhat = variational_movierating(Y);
            mse_variational(i) = mse_variational(i) + norm(Xhat - X,'fro')^2;
            
            %Bayesian PCA
            Xhat = bayesian_pca(Y);
            mse_bpca(i) = mse_bpca(i) + norm(Xhat - X,'fro')^2;
            
            %Probabilistic Matrix Factorization
            Xhat = prob_matrix_fact(Y);
            mse_pmf(i) = mse_pmf(i) + norm(Xhat - X,'fro')^2;
            
            %Weighted Trace Norm
            Xhat = weighted_trace_norm(Y,lambda);
            mse_wtn(i) = mse_wtn(i) + norm(Xhat - X,'fro')^2;
        end
    end
end

%Compute NMSE
nmse_vb = mse_vb./x_norms;
nmse_rvmlog2 = mse_rvmlog2./x_norms;
nmse_rvmschatten = mse_rvmschatten./x_norms;
nmse_nuclear = mse_nuclear./x_norms;
nmse_schatten1 = mse_schatten1./x_norms;
nmse_variational = mse_variational./x_norms;
nmse_bpca = mse_bpca./x_norms;
nmse_pmf = mse_pmf./x_norms;
nmse_wtn = mse_wtn./x_norms;

%Plot results
figure;
hold on;
plot(r_list,10*log10(nmse_vb),'-ob','linewidth',2);
plot(r_list,10*log10(nmse_rvmlog2),'-sb','linewidth',2);
plot(r_list,10*log10(nmse_rvmschatten),'-sk','linewidth',2);
plot(r_list,10*log10(nmse_nuclear),'-dc','linewidth',2);
plot(r_list,10*log10(nmse_schatten1),'-<g','linewidth',2);
plot(r_list,10*log10(nmse_variational),'-*g','linewidth',2);
plot(r_list,10*log10(nmse_bpca),'-<c','linewidth',2);
plot(r_list,10*log10(nmse_pmf),'-sr','linewidth',2);
plot(r_list,10*log10(nmse_wtn),'-or','linewidth',2);
legend('VB-1','RSVM-SN',['RSVM-SN ' num2str(s) '-norm'],'Nuclear norm','Type-1 Schatten norm','VB-2','Bayesian PCA','PMF','WTN','Location','Best');
ylabel('NMSE [dB]');
xlabel('Rank');
box on;
xlim([min(r_list) max(r_list)]);

myfontname = 'Arial';
set(gca,'FontSize',9,'fontName',myfontname);%'fontWeight','bold',
set(findall(gcf,'type','text'),'FontSize',9,'fontName',myfontname);%'fontWeight','bold',
lineobj = findobj('type', 'line');
set(lineobj, 'linewidth', 1.8);

%Save results
save([filename '.mat'],'p','q','s','r_list','alpha','M','maxiter','SNR','nmse_vb','nmse_rvmlog2','nmse_rvmschatten','nmse_nuclear','nmse_schatten1','nmse_variational','nmse_bpca','nmse_pmf','nmse_wtn');

toc;
