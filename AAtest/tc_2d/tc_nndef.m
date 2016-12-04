function [ip_new]=tc_nndef(ip)

ip_new=ip;
ip_new(ip==3)=0;
ip_new(ip==0)=1;
ip_new(ip==1)=0;
ip_new(ip==2)=10;

end