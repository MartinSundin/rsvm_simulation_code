%Plot of NMSE vs. q for Matrix completion
%Martin Sundin, 2014-11-11
tic;

%Data is saved in filename.mat
filename = 'q_comp1_test1';

p = 15;%Height of X
q_list = 3:2:30%Width of X
r = 3;%Rank of X
s = 0.5;%s value used by RSVM-SN

alpha = 0.7;%alpha = m/pq
M = 10;%Number of sensing matrix realizations
maxiter = 10;%Number of X matrix realizations
la = length(q_list);
SNR = 20;%SNR in dB

%Lists with errors
mse_vb = zeros(la,1);
mse_rvmlog_double = zeros(la,1);
mse_rvmschatten_double = zeros(la,1);
mse_nuclear = zeros(la,1);
x_norms = zeros(la,1);


parfor i = 1:la
    q = q_list(i);
    m = round(p*q*alpha);
    disp(['q = ' num2str(q)]);
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
            
            %VB
            Xhat = vb_completion(y,A,p,q,min(p,q));
            mse_vb(i) = mse_vb(i) + norm(Xhat - X,'fro')^2;
            
            %Schatten
            Xhat = rsvm_schatten(y,A,p,q,s);
            mse_rvmschatten_double(i) = mse_rvmschatten_double(i) + norm(Xhat - X,'fro')^2;
            
            %RSVM
            Xhat = rsvm_ld(y,A,p,q);
            mse_rvmlog_double(i) = mse_rvmlog_double(i) + norm(Xhat - X,'fro')^2;
            
            %Nuclear norm
            lambda = sigman*sqrt(m+sqrt(8*m));
            Xhat = nuclear_norm(y,A,p,q,lambda);
            mse_nuclear(i) = mse_nuclear(i) + norm(Xhat - X,'fro')^2;
        end
    end
end
%Compute NMSE
nmse_vb = mse_vb./x_norms;
nmse_rvmschatten_double = mse_rvmschatten_double./x_norms;
nmse_rvmlog_double = mse_rvmlog_double./x_norms;
nmse_nuclear = mse_nuclear./x_norms;

%Save results
save([filename '.mat'],'p','r','s','q_list','M','maxiter','SNR','nmse_vb','nmse_rvmschatten_double','nmse_rvmlog_double','nmse_nuclear','x_norms');

%Plot results
figure;
hold on;
plot(q_list,10*log10(nmse_vb),'-ob','linewidth',2);
plot(q_list,10*log10(nmse_rvmlog_double),'-<g','linewidth',2);
plot(q_list,10*log10(nmse_rvmschatten_double),'-sk','linewidth',2);
plot(q_list,10*log10(nmse_nuclear),'-dc','linewidth',2);
legend('VB-1','RSVM-LD',['RSVM-S, s = ' num2str(s)],'Nuclear norm','Location','Best');
ylabel('NMSE [dB]');
xlabel('q');
xlim([min(q_list) max(q_list)]);

myfontname = 'Arial';
set(gca,'FontSize',9,'fontName',myfontname);
set(findall(gcf,'type','text'),'FontSize',9,'fontName',myfontname);
lineobj = findobj('type', 'line');
set(lineobj, 'linewidth', 1.8);

toc;
