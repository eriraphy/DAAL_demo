function [data, label] = sample_spiral(num_cluster, points_per_cluster,scale)
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
delta = 2*pi/num_cluster;
bandwidth = 0.1;
points_per_cluster=points_per_cluster(:);

data = zeros([sum(points_per_cluster), 2]);
label = zeros(sum(points_per_cluster),1);

idx = 1;

for k = 1 : num_cluster
    w = rand(points_per_cluster(k),1);
    data(idx:idx+points_per_cluster(k)-1,1) = scale*(4 * w + 1) .* cos(2 * pi * w + (k-1)*delta) + randn(points_per_cluster(k),1) * bandwidth;
    data(idx:idx+points_per_cluster(k)-1,2) = scale*(4 * w + 1) .* sin(2 * pi * w + (k-1)*delta) + randn(points_per_cluster(k),1) * bandwidth;
    label(idx:idx+points_per_cluster(k)-1)=k;
    idx = idx + points_per_cluster(k);
end

