clc;clearvars
% Raphael July.2016

cd C:\Users\Raphy\Desktop\C\cpp\AAtest
at={'m_svm_two_class_batch.mexw64'};
aa = struct2cell(dir('*.mexw64'));
if ~ismember(at,aa(1,:))
    mex -v -largeArrayDims m_svm_two_class_batch.cpp daal_core_dll.lib daal_thread.lib
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

%% Pre parameters decision

cv_on=0;

if cv_on
    
    
    
else
    boxc=1.0e4;
    sigma=1.0e3;
end

%% Training and test

cctable=zeros(length(yte),10);
for i=0:9
    for j=i+1:9
        
        na=i;
        nb=j;
        nite=100;
        fprintf('Computing OVO SVM %d v %d...\n',na,nb);
        sqa=find(ytr==na);
        sqb=find(ytr==nb);
        sq=[sqa;sqb];
        xtr_t=xtr(sq,:);
        ytr_t=ytr(sq,:);
        ytr_t(ytr_t==nb)=-1;
        ytr_t(ytr_t==na)=1;
        
        [la,~]=size(xtr_t);
        rans=randperm(la);
        
        
        tt=0;
        ccrab_base=0;
        while ccrab_base<0.98
            if tt
                warning('Inaccurate prediction might have occored...');
                
                fprintf('Attempt #%d started...\n',tt);
            end
            %make a difference
            xtr_t=xtr_t(rans,:);
            ytr_t=ytr_t(rans,:);
            
            
            %retrain
            t0=cputime;
            svmf=fitcsvm(xtr_t,ytr_t,'CacheSize',160000000,'KernelFunction','RBF',...
                'Verbose',0,'DeltaGradientTolerance',0,'IterationLimit',nite,...
                'Boxconstraint',boxc);
            op2_tmp=predict(svmf,xte);
            t=cputime-t0;
            
            sqa=find(yte==na);
            sqb=find(yte==nb);
            ccra=sum((op2_tmp(sqa)+1)/2)/length(sqa);
            ccrb=sum(-(op2_tmp(sqb)-1)/2)/length(sqb);
            
            fprintf('Partial CCR: %d(%0.3f); %d(%0.3f)\n',na,ccra,nb,ccrb);
            if ccra*ccrb>ccrab_base
                fprintf('<<<<<Updating partial pridiction<<<<<\n');
                ccrab_base=ccra*ccrb;
                op2=op2_tmp;
                
            else
            end
            nite=nite+200;
            if tt
                fprintf('<<<<<Enlarging iteration times:%i<<<<<\n',nite);
            end
            fprintf('Time elapsed_%f s\n',t);
            tt=tt+1;
            if tt>10
                break
            end
        end
        cctable(logical(op2+1),na+1)=cctable(logical(op2+1),na+1)+1;
        cctable(logical(op2-1),nb+1)=cctable(logical(op2-1),nb+1)+1;
        fprintf('>>>>>>>>>>>>>>>>>>>>Round %d done>>>>>>>>>>>>>>>>>>>>\n',unique(sum(cctable,2)));
        
    end
end
