%Plot of NMSE vs. alpha = m/p*q for Matrix reconstruction and different VB
%methods
%Martin Sundin, 2014-11-27
tic;


%Data is saved in filename.mat
filename = 'alpha_vb_rec1_test1';

p = 25;%Height of X
q = 50;%Width of X
r = 3;%Rank of X

alpha_list = 0.3:0.1:0.9
M = 20;%Number of sensing matrix realizations
maxiter = 10;%Number of X matrix realizations
la = length(alpha_list);
SNR = 20;%SNR in dB

%Lists with errors
mse_nuclear = zeros(la,1);
mse_vb = zeros(la,1);
mse_vb_col = zeros(la,1);
mse_vb_row = zeros(la,1);
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
            
            %Nuclear norm
            lambda = sigman*sqrt(m+sqrt(8*m));
            Xhat = nuclear_norm(y,A,p,q,lambda);
            mse_nuclear(i) = mse_nuclear(i) + norm(Xhat - X,'fro')^2;
            
            %VB reconstruction
            Xhat = vb_reconstruction(y,A,p,q,r);
            mse_vb(i) = mse_vb(i) + norm(Xhat - X,'fro')^2;
            
            %VB column reconstruction
            Xhat = vb_reconstruction_column(y,A,p,q,r);
            mse_vb_col(i) = mse_vb_col(i) + norm(Xhat - X,'fro')^2;
            
            %VB row reconstruction
            Xhat = vb_reconstruction_row(y,A,p,q,r);
            mse_vb_row(i) = mse_vb_row(i) + norm(Xhat - X,'fro')^2;
        end
    end
end

%Compute NMSE
nmse_nuclear = mse_nuclear./x_norms;
nmse_vb = mse_vb./x_norms;
nmse_vb_col = mse_vb_col./x_norms;
nmse_vb_row = mse_vb_row./x_norms;

%Save results
save([filename '.mat'],'p','q','r','alpha_list','M','maxiter','SNR','nmse_nuclear','nmse_vb','nmse_vb_col','nmse_vb_row','x_norms');

%Plot results
figure;
hold on;
plot(alpha_list,10*log10(nmse_vb),'-ob');
plot(alpha_list,10*log10(nmse_vb_col),'-*g');
plot(alpha_list,10*log10(nmse_vb_row),'-sk');
plot(alpha_list,10*log10(nmse_nuclear),'-dc');
legend('VB','VB column','VB row','Nuclear norm','Location','Best');
ylabel('NMSE [dB]');
xlabel('m/pq');
xlim([min(alpha_list) max(alpha_list)]);
box on;

myfontname = 'Arial';
set(gca,'FontSize',9,'fontName',myfontname);%'fontWeight','bold',
set(findall(gcf,'type','text'),'FontSize',9,'fontName',myfontname);%'fontWeight','bold',
lineobj = findobj('type', 'line');
set(lineobj, 'linewidth', 1.8);

toc;
