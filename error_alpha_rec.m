%NMSE vs. alpha = m/p*q
%Matrix reconstruction
%Martin Sundin, 2014-11-27
tic;

%Data is saved in filename.mat
filename = 'alpha_rec1_test1';

p = 15;%Height of X
q = 30;%Width of X
r = 3;%Rank of X
s = 0.5;%s value used by RSVM-SN

alpha_list = 0.3:0.1:0.9
M = 10;%Number of sensing matrix realizations
maxiter = 10;%Number of X matrix realizations
la = length(alpha_list);
SNR = 20;%SNR in dB

%Lists with errors
mse_rvmlog_double = zeros(la,1);
mse_rvmschatten_double = zeros(la,1);
mse_nuclear = zeros(la,1);
mse_vb = zeros(la,1);
mse_schatten1 = zeros(la,1);
x_norms = zeros(la,1);

parfor i = 1:la%parfor
    m = round(p*q*alpha_list(i));
    disp(['alpha = ' num2str(alpha_list(i))]);
    sigma2n = p*q*r/m*10^(-SNR/10);
    sigman = sqrt(sigma2n);
    for m1 = 1:M
        %Generate sensing matrix
        A = randn(m,p*q);
        A = A*diag(1./sqrt(diag(A'*A)));
        for iter = 1:maxiter
            %Generate low-rank matrix X
            L = randn(p,r);
            R = randn(r,q);
            X = L*R;
            x_norms(i) = x_norms(i) + norm(X,'fro')^2;
            
            %Generate measurements
            y = A*X(:) + sigman*randn(m,1);
            
            %RSVM-SN
            %disp('RSVM-SN');
            Xhat = rsvm_schatten(y,A,p,q,s);
            mse_rvmschatten_double(i) = mse_rvmschatten_double(i) + norm(Xhat - X,'fro')^2;
            
            %RSVM-LD
            %disp('RSVM-LD');
            Xhat = rsvm_ld(y,A,p,q);
            mse_rvmlog_double(i) = mse_rvmlog_double(i) + norm(Xhat - X,'fro')^2;
            
            %Nuclear norm
            %disp('Nuclear norm');
            lambda = sigman*sqrt(m+sqrt(8*m));
            Xhat = nuclear_norm(y,A,p,q,lambda);
            mse_nuclear(i) = mse_nuclear(i) + norm(Xhat - X,'fro')^2;
            
            %Type-I Schatten
            %disp('Type-I Schatten');
            sna_param = 1e-2;
            options = optimset('GradObj', 'on', 'MaxIter', 100,'Display','off');
            Xhat = fminunc(@(t)(schatten_norm_type1(t,A,y,p,q,s,sna_param)),pinv(A)*y,options);
            Xhat = reshape(Xhat,p,q);
            mse_schatten1(i) = mse_schatten1(i) + norm(Xhat - X,'fro')^2;
            
            %VB-1 reconstruction
            %disp('VB-1');
            Xhat = vb_reconstruction(y,A,p,q,r);
            %Xhat = vb_reconstruction_row(y,A,p,q,r);
            mse_vb(i) = mse_vb(i) + norm(Xhat - X,'fro')^2;
        end
    end
end

%Compute NMSE
nmse_rvmschatten_double = mse_rvmschatten_double./x_norms;
nmse_rvmlog_double = mse_rvmlog_double./x_norms;
nmse_nuclear = mse_nuclear./x_norms;
nmse_vb = mse_vb./x_norms;
nmse_schatten1 = mse_schatten1./x_norms;

%Save results
save([filename '.mat'],'p','q','r','s','alpha_list','M','maxiter','SNR','nmse_rvmschatten_double','nmse_rvmlog_double','nmse_nuclear','nmse_vb','nmse_schatten1','x_norms');

%Plot results
figure;
hold on;
plot(alpha_list,10*log10(nmse_vb),'-ob');
plot(alpha_list,10*log10(nmse_rvmlog_double),'-sb');
plot(alpha_list,10*log10(nmse_rvmschatten_double),'-sk');
plot(alpha_list,10*log10(nmse_nuclear),'-dc');
plot(alpha_list,10*log10(nmse_schatten1),'-<g');
%legend('VB-1','RSVM-LD',['RSVM-S, s = ' num2str(s)],'Nuclear norm','Type-1 Schatten norm','Location','Best');
ylabel('NMSE [dB]');
xlabel('m/pq');
xlim([min(alpha_list) max(alpha_list)]);
box on;

myfontname = 'Arial';
set(gca,'FontSize',9,'fontName',myfontname);
set(findall(gcf,'type','text'),'FontSize',9,'fontName',myfontname);
lineobj = findobj('type', 'line');
set(lineobj, 'linewidth', 1.8);

toc;
