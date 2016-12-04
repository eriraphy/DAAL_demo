clear;clc;
% Raphael July.2016

%% kmeans

cd C:\Users\Raphy\Desktop\C\cpp\AAtest

mex -v m_kmeans_batch.cpp daal_core.lib daal_thread.lib

%xinput=importdata('..\..\data\batch\kmeans_dense.csv');


for i=10
    for j=1
        %% Parameters setting
        nClusters = i*2;
        nIterations = 20;
        accuracyT= 0.00;
        dsp=1;
        
        nptsq=j*10000*ones(1,nClusters);
        [xinput,~,ct] = sample_radiant(nClusters,nptsq,0.7,0.8,0.1);
        % [xinput,label] = sample_circle( nClusters, nptsq );
        % [xinput,label] = sample_spiral( nClusters, nptsq );
        % plot(xinput(:,1),xinput(:,2),'.');
        
        
        % delete(gcp('nocreate'))
        % parpool;
        
        %% Computation
        fprintf('Start computing...(Matlab)\n');
        t0=cputime;
        tic;
        
        [op_mtlb,ct_mtlb]=kmeans(xinput,nClusters,'Distance','sqeuclidean',...
            'Start','sample','replicates',1,'display','final','MaxIter',nIterations);
        t=cputime-t0;
        wt_mtlb(i,j)=toc;
        fprintf('Wall Time elapsed_%f s\n',toc);
        fprintf('CPU Time elapsed_%f s\n',t);
        
        fprintf('Start computing...(DAAL)\n');
        t0=cputime;
        tic;
        
        %%random sample start
        %[op1,op2,op3]=m_kmeans_batch(xinput,nClusters,nIterations,accuracyT);
        
        %given centriods start
        [op1,op2,op3]=m_kmeans_batch(xinput,nClusters,nIterations,accuracyT,ct);
        t=cputime-t0;
        wt_daal(i,j)=toc;
        fprintf('Wall Time elapsed_%f s\n',toc);
        fprintf('CPU Time elapsed_%f s\n',t);
        
        fprintf('Num of Iterations: %i\n',op3);
        fprintf('Time elapsed_%f s\n',t);
        
    end
end

%% CCR determination & plot
op_mtlb=op_mtlb-1;
turelabels_daal=0;
turelabels_mtlb=0;

k=1;
for i=1:nClusters
    temp=op1(k:k+nptsq(i)-1);
    err=temp-mode(temp);
    turelabels_daal=turelabels_daal+length(find(err==0));
    
    temp=op_mtlb(k:k+nptsq(i)-1);
    err=temp-mode(temp);
    turelabels_mtlb=turelabels_mtlb+length(find(err==0));
    k=k+nptsq(i);
end
ccr_daal=turelabels_daal/sum(nptsq);
ccr_mtlb=turelabels_mtlb/sum(nptsq);

if dsp
    figure1 = figure('Color',[1 1 1]);
    csq=colsq(nClusters);
      
    
    figure(1)
    for i=1:nClusters
        tempsq=find(op_mtlb==i-1);
        plot(xinput(tempsq,1),xinput(tempsq,2),'.','color',csq(i,:));
        hold on
    end
    plot(ct_mtlb(:,1),ct_mtlb(:,2),'xk')
    %plot(ct(:,1),ct(:,2),'xk')
    hold off
    
    figure(2)
    for i=1:nClusters
        tempsq=find(op1==i-1);
        plot(xinput(tempsq,1),xinput(tempsq,2),'.','color',csq(i,:));
        hold on
    end
    plot(op2(:,1),op2(:,2),'xk')
    hold off
    
end