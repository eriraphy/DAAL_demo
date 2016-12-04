function [stepsq,dis]=tcdis(lm,snsq,bsq)
% Raphael.May.16
% ipmat
% 0: background
% 1: snake pos
% 2: beam pos
% 3: snakehead pos
%
%
% snsq: snake pos sq
% (head to end)
% bsq: beam pos sq
%
lsn=size(snsq,1);
lb=size(bsq,1);
sq=[];
for ia=-1:1
    for ib=-1:1
        for ic=-1:1
            for i=1:lb
            sq=[sq;bsq(i,:)+lm*[ia ib ic]];
            end
        end
    end
end
spmat=zeros(3*lm,3*lm,3*lm);
snhd=snsq(1,:)+lm;
for i=1:lsn
    spmat(snsq(i,1)+lm,snsq(i,2)+lm,snsq(i,3)+lm)=1;
end
for i=1:length(sq)
    spmat(sq(i,1)+lm,sq(i,2)+lm,sq(i,3)+lm)=2;
end
for i=1:length(sq)
    bpos=sq(i,:)+lm;
    cmat=sort([snhd;bpos]);
    rtmat=spmat(cmat(1,1):cmat(2,1),cmat(1,2):cmat(2,2),cmat(1,3):cmat(2,3));
    
end