function errimgdsp(yp,yte,yerr,xte,kn,km)
% Raphael July.2016

figure1=figure('color',[1 1 1]);
%figure(1);
set(gcf, 'position', [100 50 1000 700]);

title('Misclassification');
errlist=find(yerr);
for i=1:min(kn*km,length(errlist))
    subplot(kn,km,i)
    
    samplei=errlist(randi(length(errlist)));
    errlist(errlist==samplei)=[];
    imshow(reshape(xte(samplei,:),28,28)',[0,255],'InitialMagnification','fit')
    title([num2str(yte(samplei)) '->' num2str(yp(samplei))])
    hold on
    
end
hold off