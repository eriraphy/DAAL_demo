function sq=colsq(numcol)
% Raphael July.2016

deg=2^(fix(log2(numcol))+2);
c=colormap(hsv(deg));

sq=zeros(numcol,3);
for i=1:numcol
    sq(i,:)=c(ceil(i*deg/numcol),:);
end