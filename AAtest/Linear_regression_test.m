clc;clearvars
% Raphael July.2016
cd C:\Users\Raphy\Desktop\C\cpp\AAtest
%at={'m_linear_regression_norm_eq_batch.mexw64'};
%aa = struct2cell(dir('*.mexw64'));
%if ~ismember(at,aa(1,:))
mex -v m_linear_regression_train.cpp daal_core.lib daal_thread.lib
% mex -v m_linear_regression_test.cpp daal_core.lib daal_thread.lib

%end
%% Preprocess
% trdata=importdata('..\..\data\batch\linear_regression_train.csv');
% tedata=importdata('..\..\data\batch\linear_regression_test.csv');
%
% xtr=trdata(:,1:10);
% ytr=trdata(:,11);
% xte=tedata(:,1:10);
% yte=tedata(:,11);
%
% clearvars -except ytr yte xtr xte
% cd C:\Users\Raphy\Desktop\C\cpp\AAtest

for i=1:10
    
    npts=100000;
    nfeatures=2*i;
    
    [xtr,ytr,coeff,bias]=sample_regression(npts,nfeatures,[0 10],[0 10],[0 10],0.1);
    
    xte=xtr;
    yte=ytr;
    
    fprintf('Ready to compute...\nPress any key to continue>>\n');
    % pause
    
    %%
    rptimes=100;
    
    fprintf('Start computing...(Matlab)\n');
    t0=cputime;
    tic;
    
    for j=1:rptimes
        [w,~,b]=regress(ytr,xtr);
    end
    
    t=cputime-t0;
    wt_mtlb(i)=toc;
    fprintf('Wall Time elapsed_%f s\n',toc);
    fprintf('CPU Time elapsed_%f s\n',t);
    
    fprintf('Start computing...(DAAL)\n');
    t0=cputime;
    tic;
    for j=1:rptimes
        [opw,opb]=m_linear_regression_train(xtr,ytr);
    end
    
    t=cputime-t0;
    wt_daal(i)=toc;
    fprintf('Wall Time elapsed_%f s\n',toc);
    fprintf('CPU Time elapsed_%f s\n',t);
    
    
end

yp=zeros(length(yte),1);
yop=zeros(length(yte),1);
for i=1:length(yte)
    yp(i)=xte(i,:)*w+mean(b);
    yop(i)=xte(i,:)*opw+opb;
end

mse_mtlb=mean((yp-yte).^2);
mse_daal=mean((yop-yte).^2);