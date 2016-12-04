function [ip]=tc_nndef(ip)

ip(ip==3)=0;
ip(ip==1)=-1;
ip(ip==2)=5;

end