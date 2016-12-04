clc;clearvars
% Raphael July.2016

cd C:\Users\Raphy\Desktop\C\cpp\AAtest
at={'m_svm_multi_class_batch.mexw64'};
aa = struct2cell(dir('*.mexw64'));
if ~ismember(at,aa(1,:))
    mex -v -largeArrayDims m_svm_multi_class_batch.cpp daal_core_dll.lib daal_thread.lib
end
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
    accu=0.5;
    niter_ini=50;
end

%% Training and test

niter=niter_ini;

[la,~]=size(xtr);
rans=randperm(la);
xtr=xtr(rans,:);
ytr=ytr(rans,:);

nclass=length(unique(ytr));
fprintf('Start computation...\n');

t1=cputime;
[op]=m_svm_multi_class_batch(xtr,ytr,xte,yte,nclass,niter,cbox,accu,'rbf',sigma);

t=cputime-t1;
fprintf('Process done!\n');
fprintf('Total time elapsed_%f s\n',t);

% clearvars -except ytr yte xtr xte cctable errdsp
% [pval,ppos]=sort(cctable,2,'descend');
% ppos=ppos-1;
% yp=ppos(:,1);
% confm=confusionmat(yte,yp);
% ccr=0;
% ccr_t2=0;
% ccr_t5=0;
% for i=1:length(yte)
%     ccr=ccr+ismember(yte(i),ppos(i,1))/length(yte);
%     ccr_t2=ccr_t2+ismember(yte(i),ppos(i,1:2))/length(yte);
%     ccr_t5=ccr_t5+ismember(yte(i),ppos(i,1:5))/length(yte);
% end
% yerr=and(logical(ppos(:,1)-yte),(logical(ppos(:,2)-yte)));

% if errdsp
%     set(gcf, 'position', [100 50 1000 700]);
%     kn=6;
%     km=10;
%     figure(1)
%     errlist=find(yerr);
%     for i=1:min(kn*km,length(errlist))
%         subplot(kn,km,i)
%
%         samplei=errlist(randi(length(errlist)));
%         errlist(errlist==samplei)=[];
%         imshow(reshape(xte(samplei,:),28,28)',[0,255],'InitialMagnification','fit')
%         title([num2str(yte(samplei)) '->' num2str(yp(samplei))])
%         hold on
%
%     end
%     hold off
% end

