function [xtr,ytr,coeff,bias]=sample_regression(num_pts,num_ftrs,coeff_int,bias_int,x_int,variance)

% Raphael Aug.2016

if nargin<3
    coeff_int=[-10,10];
end

if nargin<4
    bias_int=[-10,10];
end

if nargin<5
    x_int=[-10,10];
end

if nargin<6
    variance=1;
end

%coeff=zeros(1:num_ftrs);

if length(coeff_int)==1;
    coeff=(rand(1,num_ftrs)-1/2)*coeff_int*2;
    
elseif length(coeff_int)==2;
    coeff=rand(1,num_ftrs)*(coeff_int(2)-coeff_int(1))+coeff_int(1);
    
else
    error('Unexpected length of coeffient interval');
end

if length(bias_int)==1;
    bias=(rand-1/2)*bias_int*2;
    
elseif length(bias_int)==2;
    bias=rand*(bias_int(2)-bias_int(1))+bias_int(1);
    
else
    error('Unexpected length of bias interval');
end

xtr=zeros(num_pts,num_ftrs);
ytr=zeros(num_pts,1);

if length(x_int)==1;
    
    for i=1:num_pts
        x_tmp=(rand(1,num_ftrs)-1/2)*x_int*2;
        ytr(i)=x_tmp*coeff'+bias;
        for j=1:num_ftrs
            xtr(i,j)=x_tmp(j)+randn*variance;
        end
    end
    
elseif length(x_int)==2
    
    for i=1:num_pts
        x_tmp=rand(1,num_ftrs)*(x_int(2)-x_int(1))+x_int(1);
        ytr(i)=x_tmp*coeff'+bias;
        for j=1:num_ftrs
            xtr(i,j)=x_tmp(j)+randn*variance;
        end
    end
    
else
    error('Unexpected length of x inteval');
end

end