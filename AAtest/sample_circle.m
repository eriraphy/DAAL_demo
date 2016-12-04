function [data ,label] = sample_circle( num_cluster, points_per_cluster,scale )
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
points_per_cluster=points_per_cluster(:);

data = zeros([sum(points_per_cluster), 2]);
label = zeros(sum(points_per_cluster),1);
idx = 1;
bandwidth = 0.1;

for k = 1 : num_cluster
    theta = 2 * pi * rand(points_per_cluster(k), 1);
    rho = k*scale + randn(points_per_cluster(k), 1) * bandwidth;
    [x, y] = pol2cart(theta, rho);
    data(idx:idx+points_per_cluster(k)-1,:) = [x, y];
    label(idx:idx+points_per_cluster(k)-1)=k;
    idx = idx + points_per_cluster(k);
end
