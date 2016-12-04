clc;clearvars
% Raphael July.2016

cd C:\Users\Raphy\Desktop\C\cpp\AAtest
%at={'m_svm_two_class_batch.mexw64'};
%aa = struct2cell(dir('*.mexw64'));
%if ~ismember(at,aa(1,:))
    mex -v -largeArrayDims m_svm_two_class_batch.cpp daal_core_dll.lib daal_thread.lib
%end
%% Preprocess
cd ..\..\data\matlab
datatrain=struct2cell(importdata('data_mnist_train.mat'));
datatest=struct2cell(importdata('data_mnist_test.mat'));
xtr=datatrain{1};
ytr=datatrain{2};
xte=datatest{1};
yte=datatest{2};
clearvars -except ytr yte xtr xte
cd ..\..\cpp\AAtest

% Cross validation
cv_on=0;
% OVO=0; OVA=1
o_mode=0;
% Print err samples
errdsp=1;

%% Pre parameters decision

if cv_on
    % Cross validation has been done as a result of default value
    % See below
    
else
    if o_mode==0
        cbox=1.0e5;
        sigma=1.0e3;
    elseif o_mode==1
        cbox=1.0e5;
        sigma=1.0e3;
    end
    % Threshold of the accuracy
    accu=0.5;
    niter_ini=200;
end

ccr_base=0.1;


na=0;
nb=1;
niter=niter_ini;
sqa=find(ytr==na);
sqb=find(ytr==nb);
sq=[sqa;sqb];
xtr_t=xtr(sq,:);
ytr_t=ytr(sq,:);
ytr_t(ytr_t==nb)=-1;
ytr_t(ytr_t==na)=1;

[la,~]=size(xtr_t);
rans=randperm(la);

xtr_t=xtr_t(rans,:);
ytr_t=ytr_t(rans,:);

xte_t=xtr_t;
yte_t=ytr_t;


C=1;
t_daal=zeros(1,25);
t_mtlb=zeros(1,25);

for i=1
    maxite=100*i;
    
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
        'Verbose',0,'DeltaGradientTolerance',0,'IterationLimit',maxite*10,...
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
