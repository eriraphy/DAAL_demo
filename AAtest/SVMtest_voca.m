clc;clearvars
% Raphael July.2016
t0=cputime;
cd C:\Users\Raphy\Desktop\C\daalexamples\cpp\AAtest
at={'m_svm_two_class_batch.mexw64'};
aa = struct2cell(dir('*.mexw64'));
if ~ismember(at,aa(1,:))
    mex -v m_svm_two_class_batch.cpp daal_core.lib daal_core_dll.lib daal_sequential.lib daal_thread.lib
end
%% Preprocess
cd ..\..\data\matlab
if ~exist('datatrain','var')
    fprintf('Loading data...\n');
    voc=importdata('vocabulary.txt');
    sl=importdata('stoplist.txt');
    nl=importdata('newsgrouplabels.txt');
    datatrain=importdata('train.data');
    datatest=importdata('test.data');
    ytr=importdata('train.label');
    yte=importdata('test.label');
    t=cputime-t0;
    fprintf('Time elapsed_%f s\n',t);
else
    fprintf('Dataset already exsit\n');
end
loa=length(unique(datatrain(:,2)));
ltr=length(ytr);
lte=length(yte);
lvoc=length(voc);
slc=ismember(voc,sl);
if ~(exist('dtr','var') && exist('dte','var'))
    at={'dtr.mat'};
    aa = struct2cell(dir('*.mat'));
    if ismember(at,aa(1,:))
        fprintf('Converted data matrix already exsit\nLoading data...\n');
        dtr=importdata('dtr.mat');
        dte=importdata('dte.mat');
        fprintf('Converted data loaded\n');
        t=cputime-t0;
        fprintf('Time elapsed_%f s\n',t);
    else
        fprintf('Converting data...\n');
        dtr=full(sparse(datatrain(:,1),datatrain(:,2),datatrain(:,3)));
        dte=full(sparse(datatest(:,1),datatest(:,2),datatest(:,3)));
        for i=1:ltr
            dtr(i,:)=dtr(i,:)./sum(dtr(i,:));
        end
        for i=1:lte
            dte(i,:)=dte(i,:)./sum(dte(i,:));
        end
        [~,vctr]=size(dtr);
        d0=zeros(ltr,lvoc-vctr);
        dtr=[dtr,d0];
        dtr(:,slc)=0;
        dte(:,slc)=0;
        save('dtr.mat','dtr','-v7.3');
        save('dte.mat','dte','-v7.3');
        fprintf('Data pre_process done\n');
        t=cputime-t0;
        fprintf('Time elapsed_%f s\n',t);
    end
else
    fprintf('Converted data already exsit\n');
end

%%

    fprintf('Defining OVO labels...\n');
    na=1;
    nb=18;
    sqa=find(ytr==na);
    sqb=find(ytr==nb);
    sq=[sqa;sqb];
    dtr_t=dtr(sq,:);
    ytr_t=ytr(sq,:);
    sqa=find(yte==na);
    sqb=find(yte==nb);
    sq=[sqa;sqb];
    dte_t=dte(sq,:);
    yte_t=yte(sq,:);
    ytr_t(ytr_t==na)=1;
    ytr_t(ytr_t==nb)=-1;
    yte_t(yte_t==na)=1;
    yte_t(yte_t==nb)=-1;
    clearvars sq sqa sqb na nb
    %
    dtr_t(:,1000:end)=[];
    dte_t(:,1000:end)=[];
    % for i=1:2
    %     dtr_t=[dtr_t;dtr_t];
    %     ytr_t=[ytr_t;ytr_t];
    % end
    %
    cd C:\Users\Raphy\Desktop\C\daalexamples\data\temp
    csvwrite('dtr.csv',dtr_t);
    csvwrite('ytr.csv',ytr_t);
    csvwrite('dte.csv',dte_t);
    csvwrite('yte.csv',yte_t);
    cd C:\Users\Raphy\Desktop\C\daalexamples\cpp\AAtest

% clearvars -except ytr yte dtr dte
fprintf('Ready to compute...\nPress any key to continue>>\n');
pause

%%
clearvars



t0=cputime;
xtr=csvread('..\..\data\temp\dtr.csv');
ytr=csvread('..\..\data\temp\ytr.csv');
xte=csvread('..\..\data\temp\dte.csv');
yte=csvread('..\..\data\temp\yte.csv');
t=cputime-t0;
fprintf('Time elapsed_%f s\n',t);
% xtr(:,1000:end)=[];
% xte(:,1000:end)=[];

fprintf('Start computing...(Matlab)\n');
t0=cputime;
svm=fitcsvm(xtr,ytr);
yp=predict(svm,xte);
err=yte-yp;
ccr_mtlb=length(find(err==0))/length(yte);
t=cputime-t0;
fprintf('Time elapsed_%f s\n',t);

fprintf('Start computing...(DAAL)\n');
t0=cputime;
[op1,op2]=m_svm_two_class_batch(xtr,ytr,xte,yte);
ccr_daal=length(find(op2==1))/length(yte);
t=cputime-t0;
fprintf('Time elapsed_%f s\n',t);


