function [tpos,ppos]=plot_pos(num,thx,thy,hei,wid,mks)
row=ceil(num/3);
col=mod(num-1,3)+1;
luy=(7-row+1)*2*hei+thy;
lux=(2*thx+mks+4*wid)*(col-1)+3*thx+mks;

tpos=[lux-mks-1.5*thx,luy-hei;lux-mks-0.5*thx,luy-hei];
ppos=zeros(8,4);
for ii=1:2
    for jj=1:4
        kk=(ii-1)*4+jj;
        ppos(kk,:)=[lux+(jj-1)*wid,luy-ii*hei,wid,hei];
    end
end
end
    