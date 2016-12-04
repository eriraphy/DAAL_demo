clc;clear;
%

cd C:\Users\Raphy\Desktop\C\cpp\AAtest\tc_2d
lm=5;
[xtr,ytr]=tc_scriptA(0,lm);
t0=cputime;
dsp=0;

fprintf('Generating script samples...\n');
k=1;
for i=1:500
    [xtr_t,ytr_t]=tc_scriptA(dsp,lm);
    idx=~ismember(xtr_t,xtr,'rows');
    xtr=[xtr;xtr_t(idx,:)];
    ytr=[ytr;ytr_t(idx,:)];
    if length(xtr)/1000>=k
        t=cputime-t0;
        fprintf('%dk samples\n',k);
        fprintf('Time elapsed_%f s\n',t);
        k=k+1;
        t0=cputime;
    end
end

xtr=tc_nndef(xtr);
% rans=randperm(length(ytr));
% xtr=xtr(rans,:);
% ytr=ytr(rans,:);
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
    if ~op1(i,yte(i))==1
        err=err+1;
    end
end
ccr=1-(err/length(yte));
fprintf('Done\n');
if unique(op1)==[0];
    fprintf('>>>Error exsits<<<\n');
end

