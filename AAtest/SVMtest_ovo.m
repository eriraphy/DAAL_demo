clc;clearvars
% Raphael July.2016

%setenv('INCLUDE',[getenv('INCLUDE') ';C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.0.072\windows\tbb\include']);


%setenv('LIB',[getenv('LIB') ';C:\Program Files (x86)\IntelSWTools\compilers_and_libraries_2017.0.072\windows\tbb\lib\intel64_win\vc14']);


cd C:\Users\Raphy\Desktop\C\cpp\AAtest
% at={'m_svm_ovo_batch.mexw64'};
% aa = struct2cell(dir('*.mexw64'));
% if ~ismember(at,aa(1,:))
mex -v -largeArrayDims m_svm_ovo_batch.cpp daal_core_dll.lib daal_thread.lib tbb.lib
% end
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
errdsp=0;

%% Pre parameters decision

if cv_on
    % Cross validation has been done as a result of default value
    % See below
    
else
    cbox=1.0e5;
    sigma=1.0e3;
    
    % Threshold of the accuracy
    accu=0.01;
    niter_ini=50;
end

%% Training and test

niter=niter_ini;

% [la,~]=size(xtr);
% rans=randperm(la);
% xtr=xtr(rans,:);
% ytr=ytr(rans,:);

nclass=length(unique(ytr));

fprintf('Start computation...\n');

% na=0;
% nb=1;
% sqa=find(ytr==na);
% sqb=find(ytr==nb);
% sq=[sqa;sqb];
% xtr_t=xtr(sq,:);
% ytr_t=ytr(sq,:);
% ytr_t(ytr_t==nb)=-1;
% ytr_t(ytr_t==na)=1;


t1=cputime;
tic;
[op]=m_svm_ovo_batch(xtr,ytr,xte,yte,nclass,niter,cbox,accu,'rbf',sigma);

t=cputime-t1;
toc;
fprintf('CPU time is %f seconds\n',t);
fprintf('Process done!\n');

clearvars -except ytr yte xtr xte op errdsp

[pval,ppos]=sort(op,2,'descend');
ppos=ppos-1;
yp=ppos(:,1);
confm=confusionmat(yte,yp);
ccr=0;
ccr_t2=0;
ccr_t5=0;
for i=1:length(yte)
    ccr=ccr+ismember(yte(i),ppos(i,1))/length(yte);
    ccr_t2=ccr_t2+ismember(yte(i),ppos(i,1:2))/length(yte);
    ccr_t5=ccr_t5+ismember(yte(i),ppos(i,1:5))/length(yte);
end


yerr=and(logical(ppos(:,1)-yte),(logical(ppos(:,2)-yte)));

if errdsp
   errimgdsp(yp,yte,yerr,xte,6,8)
end

