function [ipmat]=tcplot(lm,snsq,bsq,dsp,mode)
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
% (head to end)
% bsq: beam pos sq
% 
% dsp
% 1: display
% 0: non display
% 
% mode
% 1: normal ipmat
% 2: snhd as start of ipmat
if nargin<4
    dsp=0;
end

if nargin<5
    mode=1;
end

ipmat_temp=zeros(lm,lm,lm);
[lsn,~]=size(snsq);
[lb,~]=size(bsq);
ipmat_temp(snsq(1,1),snsq(1,2),snsq(1,3))=3; %%
for i=2:lsn
    ipmat_temp(snsq(i,1),snsq(i,2),snsq(i,3))=1;
end
for i=1:lb
    ipmat_temp(bsq(i,1),bsq(i,2),bsq(i,3))=2;
end

if mode==1;
    ipmat=ipmat_temp;
elseif mode==2
    snhd=snsq(1,:);
    for i=1:3
        ipmat_temp=cat(i,ipmat_temp,ipmat_temp);
    end
    ipmat=ipmat_temp(snhd(1):snhd(1)+lm-1,snhd(2):snhd(2)+lm-1,snhd(3):snhd(3)+lm-1);
end
if dsp==1
    set(gcf, 'position', [200 100 900 600]);
    figure(1)
    axis([0 lm+1 0 lm+1 0 lm+1])
    for i=1:lb
        plot3(bsq(i,1),bsq(i,2),bsq(i,3),'r.','markersize',20)
        hold on
    end
    for i=1:lsn-1
        if abs(sum(snsq(i,:)-snsq(i+1,:)))==1
            plot3([snsq(i,1) snsq(i+1,1)],[snsq(i,2) snsq(i+1,2)],[snsq(i,3) snsq(i+1,3)],'b-','linewidth',5)
            hold on
        else
            sqc=0.5*sign(snsq(i,:)-snsq(i+1,:));
            sqa=snsq(i,:)+sqc;
            sqb=snsq(i+1,:)-sqc;
            plot3([snsq(i,1) sqa(1)],[snsq(i,2) sqa(2)],[snsq(i,3) sqa(3)],'b-','linewidth',5)
            plot3([snsq(i+1,1) sqb(1)],[snsq(i+1,2) sqb(2)],[snsq(i+1,3) sqb(3)],'b-','linewidth',5)
            hold on
        end
        plot3(snsq(i+1,1),snsq(i+1,2),snsq(i+1,3),'b.','markersize',15)
    end
    plot3(snsq(1,1),snsq(1,2),snsq(1,3),'g.','markersize',20)
    drawnow;
    grid on
    hold off
    
    view([45 30])
end