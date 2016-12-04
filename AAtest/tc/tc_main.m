clc;clear;
%
cd C:\Users\Raphy\Desktop\C\cpp\AAtest\tc
xtr=[];
ytr=[];

fprintf('Generating script samples...\n');
for i=1:200
    [xtr_t,ytr_t]=tc_scriptA(0,4);
    xtr=[xtr;xtr_t];
    ytr=[ytr;ytr_t];
end
xtr=tc_nndef(xtr);
rans=randperm(length(ytr));
xtr=xtr(rans,:);
ytr=ytr(rans,:);
xte=xtr;
yte=ytr;


clearvars -except xtr ytr xte yte



cd C:\Users\Raphy\Desktop\C\cpp\AAtest
mex -v m_neural_network_batch.cpp daal_core_dll.lib daal_thread.lib

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

