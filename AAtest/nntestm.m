clc;clear;
% Raphael July.2016
cd C:\Users\Raphy\Desktop\C\cpp\AAtest
mex -v m_neural_network_batch.cpp daal_core_dll.lib daal_thread.lib%run at the first time

% cd ..\..\data\matlab
% datatrain=struct2cell(importdata('data_mnist_train.mat'));
% datatest=struct2cell(importdata('data_mnist_test.mat'));
% xtr=datatrain{1};
% ytr=datatrain{2};
% xte=datatest{1};
% yte=datatest{2};
% cd ..\..\cpp\AAtest

% rans=randperm(length(ytr));
% xtr=xtr(rans,:);
% ytr=ytr(rans,:);


% xtr=importdata('..\..\data\batch\neural_network_train.csv');
% ytr=importdata('..\..\data\batch\neural_network_train_ground_truth.csv');
% xte=importdata('..\..\data\batch\neural_network_test.csv');
% yte=importdata('..\..\data\batch\neural_network_test_ground_truth.csv');

xte=importdata('..\..\data\batch\neural_network_train.csv');
yte=importdata('..\..\data\batch\neural_network_train_ground_truth.csv');


clearvars -except xtr xte ytr yte
fprintf('Ready to compute...\nPress any key to continue>>\n');
%pause;


fprintf('Start computing...\n');
nclass=length(unique(ytr));
[op1,op2]=m_neural_network_batch(xtr,ytr,xte,yte,nclass);

err=0;
for i=1:length(yte)
    if ~op1(i,yte(i)+1)==1
        err=err+1;
    end
end
ccr=1-(err/length(yte));
fprintf('Done\n');
if unique(op1)==[0];
    fprintf('>>>Error exsits<<<\n');
end