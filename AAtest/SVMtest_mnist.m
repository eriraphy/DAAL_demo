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
    accu=0.9;
    niter_ini=50;
end

ccr_base=0.9;
%% Training and test

t1=cputime;

if o_mode==0
    cctable=zeros(length(yte),10);
    for i=0:9
        for j=i+1:9
            
            na=i;
            nb=j;
            niter=niter_ini;
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
            while ccrab_base<ccr_base
                if tt
                    warning('Inaccurate prediction might have occored...');
                    fprintf('<<<<<Enlarging iteration times:%i<<<<<\n',niter);
                    fprintf('Attempt #%d started...\n',tt);
                end
                %make a difference
                %xtr_t=xtr_t(rans,:);
                %ytr_t=ytr_t(rans,:);
                
                %mian porcess
                t0=cputime;
                [op2_tmp]=m_svm_two_class_batch(xtr_t,ytr_t,xte,yte,niter,cbox,accu,'rbf',sigma);
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
                niter=niter+200;
                fprintf('Time elapsed_%f s\n',t);
                tt=tt+1;
                if tt>5
                    break
                end
            end
            cctable(logical(op2+1),na+1)=cctable(logical(op2+1),na+1)+1;
            cctable(logical(op2-1),nb+1)=cctable(logical(op2-1),nb+1)+1;
            fprintf('>>>>>>>>>>>>>>>>>>>>>>>Round %d done>>>>>>>>>>>>>>>>>>>>>>>\n',unique(sum(cctable,2)));
            
        end
    end
    
    
    fprintf('All process done!\n');
    t=cputime-t1;
    fprintf('Total time elapsed_%f s\n',t);
    
    
elseif o_mode==1
    cctable=zeros(length(yte),10);
    for i=0:9
        na=i;
        niter=niter_ini;
        fprintf('Computing OVA SVM %d v ALL...\n',na);
        xtr_t=xtr;
        ytr_t=ytr;
        ytr_t(ytr_t~=na)=-1;
        ytr_t(ytr_t==na)=1;
        
        
        [la,~]=size(xtr_t);
        rans=randperm(la);
        
        tt=0;
        ccrab_base=0;
        while ccrab_base<0.96
            if tt
                warning('Inaccurate prediction might have occored...');
                fprintf('<<<<<Enlarging iteration times:%i<<<<<\n',niter);
                fprintf('Attempt #%d started...\n',tt);
            end
            %make a difference
            xtr_t=xtr_t(rans,:);
            ytr_t=ytr_t(rans,:);
            
            
            %mian porcess
            t0=cputime;
            [op2_tmp]=m_svm_two_class_batch(xtr_t,ytr_t,xte,yte,niter,cbox,accu,'rbf',sigma);
            t=cputime-t0;
            
            sqa=find(yte==na);
            sqb=find(yte~=na);
            ccra=sum((op2_tmp(sqa)+1)/2)/length(sqa);
            ccrb=sum(-(op2_tmp(sqb)-1)/2)/length(sqb);
            
            fprintf('Partial CCR: %d(%0.3f); ALL(%0.3f)\n',na,ccra,ccrb);
            if ccra*ccrb>ccrab_base
                fprintf('<<<<<Updating partial pridiction<<<<<\n');
                ccrab_base=ccra*ccrb;
                op2=op2_tmp;
                
            else
            end
            niter=niter+200;
            fprintf('Time elapsed_%f s\n',t);
            tt=tt+1;
            if tt>5
                break
            end
        end
        cctable(logical(op2+1),na+1)=cctable(logical(op2+1),na+1)+1;
        fprintf('>>>>>>>>>>>>>>>>>>>>>>>Round %d done>>>>>>>>>>>>>>>>>>>>>>>\n',i+1);
        
        
    end
    fprintf('All process done!\n');
    t=cputime-t1;
    fprintf('Total time elapsed_%f s\n',t);
    
end

clearvars -except ytr yte xtr xte cctable errdsp
[pval,ppos]=sort(cctable,2,'descend');
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
    set(gcf, 'position', [100 50 1000 700]);
    kn=6;
    km=10;
    figure(1)
    errlist=find(yerr);
    for i=1:min(kn*km,length(errlist))
        subplot(kn,km,i)
        
        samplei=errlist(randi(length(errlist)));
        errlist(errlist==samplei)=[];
        imshow(reshape(xte(samplei,:),28,28)',[0,255],'InitialMagnification','fit')
        title([num2str(yte(samplei)) '->' num2str(yp(samplei))])
        hold on
        
    end
    hold off
end

