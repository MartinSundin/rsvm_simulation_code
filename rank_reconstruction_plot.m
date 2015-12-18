%Plot NMSE vs. SNR
%Matrix reconstruction
%Martin Sundin, 2014-04-12
tic;

%Data is saved in filename.mat
filename = 'rank_reconstruction1_test1';

p = 15;%Height of X
q = 30;%Width of X
s = 0.5;%s value used by RSVM-SN
alpha = 0.7;%alpha = m/pq
r_list = 1:1:10%Rank of X

M = 10;%Number of sensing matrix realizations
maxiter = 10;%Number of X matrix realizations
SNR = 40;%20;%SNR in dB
la = length(r_list);
m = round(p*q*alpha);

%Lists with errors
mse_rvmlog = zeros(la,1);
mse_rvmschatten = zeros(la,1);
mse_nuclear = zeros(la,1);
mse_schatten1 = zeros(la,1);
mse_vb = zeros(la,1);
x_norms = zeros(la,1);

for i = 1:la
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

            %RSVM-LD
            Xhat = rsvm_ld(y,A,p,q);
            mse_rvmlog(i) = mse_rvmlog(i) + norm(Xhat - X,'fro')^2;
            
            %RSVM-SN
            Xhat = rsvm_schatten(y,A,p,q,s);
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

            %VB for reconstruction
            [Xhat,~,~] = vb_reconstruction_row(y,A,p,q,r);
            mse_vb(i) = mse_vb(i) + norm(Xhat - X,'fro')^2;
        end
    end
end

%Compute NMSE
nmse_rvmlog = mse_rvmlog./x_norms;
nmse_rvmschatten = mse_rvmschatten./x_norms;
nmse_nuclear = mse_nuclear./x_norms;
nmse_schatten1 = mse_schatten1./x_norms;
nmse_vb = mse_vb./x_norms;

%Plot results
figure;
hold on;
plot(r_list,10*log10(nmse_rvmlog),'-sb','linewidth',2);
plot(r_list,10*log10(nmse_rvmschatten),'-sk','linewidth',2);
plot(r_list,10*log10(nmse_nuclear),'-dc','linewidth',2);
plot(r_list,10*log10(nmse_schatten1),'-<g','linewidth',2);
plot(r_list,10*log10(nmse_vb),'-ob','linewidth',2);
legend('RSVM-LD',['RSVM-SN, s = ' num2str(s)],'NN','SNA','VB-1','Location','Best');
ylabel('NMSE [dB]');
xlabel('Rank');
xlim([min(r_list) max(r_list)]);
box on;

myfontname = 'Arial';
set(gca,'FontSize',9,'fontName',myfontname);%'fontWeight','bold',
set(findall(gcf,'type','text'),'FontSize',9,'fontName',myfontname);%'fontWeight','bold',
lineobj = findobj('type', 'line');
set(lineobj, 'linewidth', 1.8);

%Save results
save([filename '.mat'],'p','q','s','r_list','alpha','M','maxiter','SNR','nmse_rvmlog','nmse_rvmschatten','nmse_nuclear','nmse_schatten1','nmse_vb');

toc;
