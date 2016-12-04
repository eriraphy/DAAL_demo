% setenv('PATH',[getenv('PATH') ';C:\Users\Raphy\Desktop\C\daalexamples\cpp\source']);
clc;clear;

%% linear regression

% mex -v m_linear_regression_norm_eq_batch.cpp daal_core.lib daal_core_dll.lib daal_sequential.lib daal_thread.lib%run at the first time
% cd C:\Users\Raphy\Desktop\C\cpp\AAtest
% trdata=importdata('..\..\data\batch\linear_regression_train.csv');
% tedata=importdata('..\..\data\batch\linear_regression_test.csv');
%
%
% xtr=trdata(:,1:10);
% ytr=trdata(:,11:end);
% xte=tedata(:,1:10);
% yte=tedata(:,11:end);
%
% fprintf('Start computing...(Matlab)\n');
% t0=cputime;
% yp=zeros(length(xte),2);
% for i=1:size(ytr,2)
%     [model,~,R]=regress(ytr(:,i),xtr);
%     for j=1:length(xte)
%         yp(j,i)=xte(j,:)*model+mean(R);
%     end
% end
% t=cputime-t0;
% fprintf('Time elapsed_%f s\n',t);
%
%
% fprintf('Start computing...(DAAL)\n');
% t0=cputime;
% [op1,op2]=m_linear_regression_norm_eq_batch(xtr,ytr,xte);
% t=cputime-t0;
% fprintf('Time elapsed_%f s\n',t);



%% svm dense

mex -v -largeArrayDims m_svm_two_class_batch.cpp daal_core_dll.lib daal_thread.lib

% cd ..\..\data\matlab
% datatrain=struct2cell(importdata('data_mnist_train.mat'));
% datatest=struct2cell(importdata('data_mnist_test.mat'));
% xtr=datatrain{1};
% ytr=datatrain{2};
% xte=datatest{1};
% yte=datatest{2};
% clearvars -except ytr yte xtr xte
% cd ..\..\cpp\AAtest
%
na=0;
nb=1;
% sqa=find(ytr==na);
% sqb=find(ytr==nb);
% sq=[sqa;sqb];
% xtr_t=xtr(sq,:);
% ytr_t=ytr(sq,:);
% ytr_t(ytr_t==nb)=-1;
% ytr_t(ytr_t==na)=1;
%
% [la,~]=size(xtr_t);
% rans=randperm(la);
% xtr_t=xtr_t(rans,:);
% ytr_t=ytr_t(rans,:);

cd ..\AAtemp
xtr_t=importdata('xtr.csv');
ytr_t=importdata('ytr.csv');
xte_t=importdata('xte.csv');
yte_t=importdata('yte.csv');
cd ..\AAtest


C=1e5;
t_daal=zeros(1,25);
t_mtlb=zeros(1,25);

for i=1:1
    maxite=50*i;
    
    fprintf('Start computing...(DAAL)\n');
    t0=cputime;
    tic;
    [op]=m_svm_two_class_batch(xtr_t,ytr_t,xte_t,yte_t,maxite,C,1e-10);
    
    t=cputime-t0;
    wt_daal(i)=toc;
    fprintf('Wall Time elapsed_%f s\n',toc);
    fprintf('CPU Time elapsed_%f s\n',t);
    
    
    fprintf('Start computing...(Matlab)\n');
    t0=cputime;
    tic;

    svmf=fitcsvm(xtr_t,ytr_t,'CacheSize',160000000,'KernelFunction','linear',...
        'Verbose',0,'DeltaGradientTolerance',0,'IterationLimit',maxite,...
        'Boxconstraint',C);
    yp=predict(svmf,xte_t);
    
    t=cputime-t0;
    wt_mtlb(i)=toc;
    fprintf('Wall Time elapsed_%f s\n',toc);
    fprintf('CPU Time elapsed_%f s\n',t);
    
    na=1;
    nb=-1;
    
    sqa=find(yte_t==na);
    sqb=find(yte_t==nb);
    ccra_daal=sum((op(sqa)+1)/2)/length(sqa);
    ccrb_daal=sum(-(op(sqb)-1)/2)/length(sqb);
    ccra_matlab=sum((yp(sqa)+1)/2)/length(sqa);
    ccrb_matlab=sum(-(yp(sqb)-1)/2)/length(sqb);
    
end

% figure(1)
% plot(1:25,t_daal)
% hold on
% plot(1:25,t_mtlb)

%% pca

% mex -v m_pca_svd_batch.cpp daal_core.lib daal_core_dll.lib daal_sequential.lib daal_thread.lib
%
% t0=cputime;
% xinput=importdata('..\..\data\batch\pca_normalized.csv');
%
% %xinput=rand(100000,100);
% t=cputime-t0;
% fprintf('Time elapsed_%f s\n',t);
%
% fprintf('Start computing...(Matlab)\n');
% t0=cputime;
% [evec,~,eval]=pca(xinput);
% t=cputime-t0;
% fprintf('Time elapsed_%f s\n',t);
%
%
% fprintf('Start computing...(DAAL)\n');
% t0=cputime;
% [op1,op2]=m_pca_svd_batch(xinput);
% t=cputime-t0;
% fprintf('Time elapsed_%f s\n',t);


%% kmeans

% mex -v m_kmeans_batch.cpp daal_core.lib daal_thread.lib
%
% t0=cputime;
% %xinput=importdata('..\..\data\batch\kmeans_dense.csv');
%
% nClusters =16;
% nIterations = 20;
%
% nptsq=1000000*ones(1,nClusters);
% [xinput,~,ct] = sample_radiant(nClusters,nptsq,0.7,0.6,0.1);
% %[xinput,label] = sample_circle( nClusters, nptsq );
% %[xinput,label] = sample_spiral( nClusters, nptsq );
% %plot(xinput(:,1),xinput(:,2),'.');
%
%
% t=cputime-t0;
% fprintf('Time elapsed_%f s\n',t);
%
% fprintf('Start computing...(Matlab)\n');
% t0=cputime;
% [op_mtlb,ct_mtlb]=kmeans(xinput,nClusters,'Distance','sqeuclidean',...
%     'Start',ct,'replicates',1,'display','final');
% t=cputime-t0;
% fprintf('Time elapsed_%f s\n',t);
%
% fprintf('Pausing...\n');
% fprintf('5\n');pause(1);fprintf('4\n');pause(1);fprintf('3\n');pause(1);fprintf('2\n');pause(1);fprintf('1\n');pause(1);
%
% fprintf('Start computing...(DAAL)\n');
% t0=cputime;
% [op1,op2,op3]=m_kmeans_batch(xinput,nClusters,nIterations,ct);
% %[op1,op2,op3]=m_kmeans_batch(xinput,nClusters,nIterations);
% ta=log(op1);
% t=cputime-t0;
% fprintf('Time elapsed_%f s\n',t);
%
% op_mtlb=op_mtlb-1;
% turelabels_daal=0;
% turelabels_mtlb=0;
%
% k=1;
% for i=1:nClusters
%     temp=op1(k:k+nptsq(i)-1);
%     err=temp-mode(temp);
%     turelabels_daal=turelabels_daal+length(find(err==0));
%
%     temp=op_mtlb(k:k+nptsq(i)-1);
%     err=temp-mode(temp);
%     turelabels_mtlb=turelabels_mtlb+length(find(err==0));
%     k=k+nptsq(i);
% end
% ccr_daal=turelabels_daal/sum(nptsq);
% ccr_mtlb=turelabels_mtlb/sum(nptsq);
%
% csq=colsq(nClusters);
%
% figure(1)
% for i=1:nClusters
%     tempsq=find(op_mtlb==i-1);
%     plot(xinput(tempsq,1),xinput(tempsq,2),'.','color',csq(i,:));
%     hold on
% end
% plot(ct_mtlb(:,1),ct_mtlb(:,2),'xk')
% %plot(ct(:,1),ct(:,2),'xk')
% hold off
%
% figure(2)
% for i=1:nClusters
%     tempsq=find(op1==i-1);
%     plot(xinput(tempsq,1),xinput(tempsq,2),'.','color',csq(i,:));
%     hold on
% end
% plot(op2(:,1),op2(:,2),'xk')
% hold off

