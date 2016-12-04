function [training_data,training_label]=tc_scriptA(dsp,lm,snsq,nbs)
t0=cputime;
if nargin<1
    dsp=0;
end
if nargin<2
    lm=5;%map size
end
if nargin<3
    snsq=[randi(lm) randi(lm)];
end
if nargin<4
    nbs=1;
end


% sample:
% clc;clear;
% t0=cputime;
% snsq=[2 2;1 2];
% nbs=1;
% lm=10;
% dsp=1;


%[lsn,~]=size(snsq);
bsq=[];
for i=1:nbs
    npx=randi(lm);
    npy=randi(lm);
    ipmat=tcplot(lm,snsq,bsq);
    while ipmat(npx,npy)~=0
        npx=randi(lm);
        npy=randi(lm);
    end
    bsq=[bsq;npx npy];
end
[lb,~]=size(bsq);

% ipmat=tcplot(lm,snsq,bsq,0);
%%
% instructions
% 1: Right
% 2: Left
% 3: Backward
% 4: Forward
gg=0;
k=1;
while gg==0
    sq=[];
    for ia=-1:1
        for ib=-1:1
                for i=1:lb
                    sq=[sq;bsq(i,:)+lm*[ia ib]];
                end

        end
    end
    for i=1:lb*9
        dis(i)=sum((sq(i,:)-snsq(1,:)).^2)^0.5;
    end
    minpos=find(dis==min(dis));
    d=sq(minpos(1),:)-snsq(1,:);
    instr=[];
    i=1;
    while i<=2
        if d(i)>0
            instr=2*i;
            [~,~,gg,~]=tc(lm,snsq,bsq,instr);
        elseif d(i)<0
            instr=2*i-1;
            [~,~,gg,~]=tc(lm,snsq,bsq,instr);
        end
        if isempty(instr)==0 && gg==0
            break
        end
        i=i+1;
    end
    tt=1;
    ilist=1:4;
    while gg==1
        if isempty(ilist)
            gg=1;
            break
        end
        %it=randi(7-tt);
        it=1;
        instr=ilist(it);
        ipos= ilist==instr;
        ilist(ipos)=[];
        [~,~,gg,~]=tc(lm,snsq,bsq,instr);
        tt=tt+1;
    end
    [snsq,bsq,gg,~]=tc(lm,snsq,bsq,instr);
    training_label(k,1)=instr;
    ipmat=tcplot(lm,snsq,bsq,dsp,2);
    training_data(k,:)=reshape(ipmat,1,lm^2);
    k=k+1;
    t=cputime-t0;
    if dsp
        fprintf('\n Time elapsed_%f s\n',t);
        fprintf('%2d ',d);
        fprintf('\n %2d ',k);
    end
end
if dsp
    fprintf('\n Game Over');
end



end