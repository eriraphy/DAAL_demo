clc;clear;
% pca test
% Raphael July.2016

mex -v m_pca_svd_batch.cpp daal_core.lib daal_thread.lib

%xinput=importdata('..\..\data\batch\pca_normalized.csv');


% wt_mtlb=zeros(1,5);
% wt_daal=zeros(1,5);


for j=1:1:5;
    nf=j*20;
    xinput=rand(100000,nf);
    rptimes=10;
    
    fprintf('Start computing...(Matlab)\n');
    t0=cputime;
    tic;
    
    for i=1:rptimes
        [evec,~,eval]=pca(xinput);
    end
    
    t=cputime-t0;
    wt_mtlb(j)=toc;
    fprintf('Wall Time elapsed_%f s\n',toc);
    fprintf('CPU Time elapsed_%f s\n',t);
    
    
    fprintf('Start computing...(DAAL)\n');
    t0=cputime;
    tic;
    
    for i=1:rptimes
        [opeval,opevec]=m_pca_svd_batch(xinput);
    end
    
    t=cputime-t0;
    wt_daal(j)=toc;
    fprintf('Wall Time elapsed_%f s\n',toc);
    fprintf('CPU Time elapsed_%f s\n',t);
end

% figure(1)
% plot(wt_mtlb,'b','linewidth',4)
% hold on
% plot(wt_daal,'r','linewidth',4)