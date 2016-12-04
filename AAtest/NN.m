clc;clear;
% Raphael July.2016

cd C:\Users\Raphy\Desktop\C\cpp\AAtest
mex -v m_neural_network.cpp daal_core_dll.lib daal_thread.lib
cd ..\..\data\matlab
datatrain=struct2cell(importdata('data_mnist_train.mat'));
datatest=struct2cell(importdata('data_mnist_test.mat'));
xtr=datatrain{1};
ytr=datatrain{2};
xte=datatest{1};
yte=datatest{2};
cd ..\..\cpp\AAtest

rans=randperm(length(ytr));
xtr=xtr(rans,:);
ytr=ytr(rans,:);

xtr(10001:end,:)=[];
ytr(10001:end,:)=[];


clearvars -except xtr xte ytr yte
fprintf('Ready to compute...\nPress any key to continue>>\n');
%pause;


fprintf('Start computing...\n');
nclass=length(unique(ytr));
[op1,op2]=m_neural_network(xtr,ytr,xte,yte,nclass);