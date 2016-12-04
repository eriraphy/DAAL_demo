function [snsq,bsq,gg,ipmat]=tc(lm,snsq,bsq,instr)
% Raphael.May.16

% ipmat
% 0: background
% 1: snake pos
% 2: beam pos
% 3: snakehead pos
%
% instructions
% 1: Right
% 2: Left
% 3: Backward
% 4: Forward
% 5: Down
% 6: Up
%
% snsq: snake pos sq
% bsq: beam pos sq

% samp
% snsq=[2 2;1 2];
% bsq=[1 1];
% lm=5;

gg=0;
ipmat=tcplot(lm,snsq,bsq);
[lb,~]=size(bsq);
sh=snsq(1,:);
switch(instr)
    case{1}
        sh(1)=sh(1)-1;
    case{2}
        sh(1)=sh(1)+1;
    case{3}
        sh(2)=sh(2)-1;
    case{4}
        sh(2)=sh(2)+1;
end

if sh(1)>lm
    sh(1)=sh(1)-lm;
end
if sh(1)<1
    sh(1)=sh(1)+lm;
end
if sh(2)>lm
    sh(2)=sh(2)-lm;
end
if sh(2)<1
    sh(2)=sh(2)+lm;
end

if ipmat(sh(1),sh(2))==0
    snsq=[sh;snsq(1:end-1,:)];
elseif ipmat(sh(1),sh(2))==2
    snsq=[sh;snsq];
    
    for i=1:lb
        if bsq(i,:)==sh
            pos=i;
        end
    end
    bsq(pos,:)=[];
    
    [npx,npy]=find(ipmat==0);
    if isempty(npx)
        gg=1;
    else
        rd=randi(length(npx));
        px=npx(rd);
        py=npy(rd);
        bsq=[bsq;px py];
    end
    
    %     npx=randi(lm);
    %     npy=randi(lm);
    %     while ipmat(npx,npy)~=0
    %         npx=randi(lm);
    %         npy=randi(lm);
    %     end
    %     bsq=[bsq;npx npy];
    
    
elseif ipmat(sh(1),sh(2))==1
    gg=1;
end

ipmat=tcplot(lm,snsq,bsq);

end
