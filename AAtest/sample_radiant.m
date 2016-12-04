function [data, label,ct] = sample_radiant(num_cluster,...
    points_per_cluster,scale,clusterwidth,bandwidth)
% Raphael July.2016
if nargin <1
    num_cluster = 2;
end
if nargin <2
    points_per_cluster = 100*num_cluster*ones(num_cluster,1);
end
if nargin <3
    scale=num_cluster/2;
end
if nargin <4
    clusterwidth=0.6;
end
if nargin <5
    bandwidth = 0.1;
end

delta = 2*pi/num_cluster;
data = zeros([sum(points_per_cluster), 2]);
label = zeros(sum(points_per_cluster),1);

ct=zeros(num_cluster,2);
idx = 1;



for k = 1 : num_cluster
    theta = (delta*k) + clusterwidth*delta*rand(points_per_cluster(k), 1);
    rho = scale + randn(points_per_cluster(k), 1) * bandwidth;
    [x, y] = pol2cart(theta, rho);
    data(idx:idx+points_per_cluster(k)-1,:) = [x, y];
    label(idx:idx+points_per_cluster(k)-1)=k;
    idx = idx + points_per_cluster(k);
    
    [x,y]=pol2cart(delta*k+clusterwidth*delta/2,scale+bandwidth/2);
    ct(k,:)=[x,y];
end

