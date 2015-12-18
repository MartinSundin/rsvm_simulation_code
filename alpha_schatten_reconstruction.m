%Plot of NMSE vs. alpha = m/(p*q)
%Matrix reconstruction with RSVM-SN for different s
%Martin Sundin, 2014-06-29
tic;

%Data is saved in filename.mat
filename = 'alpha_schatten_reconstruction1_test1';

p = 15;%Height of X
q = 30;%Width of X
r = 3;%Rank of X

%Schatten s-norms
s1 = 0.1;
s2 = 0.3;
s3 = 0.5;
s4 = 0.7;
s5 = 1;

alpha_list = 0.3:0.1:0.9
M = 10;%Number of sensing matrix realizations
maxiter = 10;%Number of X matrix realizations
la = length(alpha_list);
SNR = 20;%SNR in dB

%Lists with error and cputime
mse_s1 = zeros(la,1);
mse_s2 = zeros(la,1);
mse_s3 = zeros(la,1);
mse_s4 = zeros(la,1);
mse_s5 = zeros(la,1);
x_norms = zeros(la,1);

parfor i = 1:la
    m = round(p*q*alpha_list(i));
    disp(['alpha = ' num2str(alpha_list(i))]);
    sigma2n = r*p*q/m*10^(-SNR/10);
    sigman = sqrt(sigma2n);
    for m1 = 1:M
        %Generate sensing matrix
        A = randn(m,p*q);
        A = A*diag(1./sqrt(diag(A'*A)));
        for iter = 1:maxiter
            %Generate low-rank matrix X
            X = randn(p,r)*randn(r,q);
            x_norms(i) = x_norms(i) + norm(X,'fro')^2;
            
            %Generate measurements
            y = A*X(:) + sigman*randn(m,1);
            
            %s1
            Xhat = rvm_schatten(y,A,p,q,s1);
            mse_s1(i) = mse_s1(i) + norm(Xhat - X,'fro')^2;
            
            %s2
            Xhat = rvm_schatten(y,A,p,q,s2);
            mse_s2(i) = mse_s2(i) + norm(Xhat - X,'fro')^2;
            
            %s3
            Xhat = rvm_schatten(y,A,p,q,s3);
            mse_s3(i) = mse_s3(i) + norm(Xhat - X,'fro')^2;
            
            %s4
            Xhat = rvm_schatten(y,A,p,q,s4);
            mse_s4(i) = mse_s4(i) + norm(Xhat - X,'fro')^2;
            
            %s5
            Xhat = rvm_schatten(y,A,p,q,s1);
            mse_s5(i) = mse_s5(i) + norm(Xhat - X,'fro')^2;
        end
    end
end

%Compute NMSE
nmse_s1 = mse_s1./x_norms;
nmse_s2 = mse_s2./x_norms;
nmse_s3 = mse_s3./x_norms;
nmse_s4 = mse_s4./x_norms;
nmse_s5 = mse_s5./x_norms;

%Plot results
figure;
hold on;
plot(alpha_list,10*log10(nmse_s1),'-*b','linewidth',2);
plot(alpha_list,10*log10(nmse_s2),'-sg','linewidth',2);
plot(alpha_list,10*log10(nmse_s3),'-+k','linewidth',2);
plot(alpha_list,10*log10(nmse_s4),'-dc','linewidth',2);
plot(alpha_list,10*log10(nmse_s5),'-vm','linewidth',2);
legend(['Schatten ' num2str(s1) '-norm'],['Schatten ' num2str(s2) '-norm'],['Schatten ' num2str(s3) '-norm'],['Schatten ' num2str(s4) '-norm'],['Schatten ' num2str(s5) '-norm'],'Location','Best');
ylabel('NMSE [dB]');
xlabel('m/pq');
xlim([min(alpha_list) max(alpha_list)]);

myfontname = 'Arial';
set(gca,'FontSize',9,'fontName',myfontname);%'fontWeight','bold',
set(findall(gcf,'type','text'),'FontSize',9,'fontName',myfontname);%'fontWeight','bold',
lineobj = findobj('type', 'line');
set(lineobj, 'linewidth', 1.8);

%Save results
save([filename '.mat'],'p','q','r','alpha_list','M','maxiter','SNR','s1','s2','s3','s4','s5','nmse_s1','nmse_s2','nmse_s3','nmse_s4','nmse_s5');

toc;
