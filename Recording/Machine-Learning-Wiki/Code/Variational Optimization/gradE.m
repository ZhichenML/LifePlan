function [g,G]=gradE(W,x,y)
G=repmat(W*x-y,size(W,2),1).*x;
g=sum(G,2)/size(G,2);
g=g';