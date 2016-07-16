function [assignments]=leastdistance(descriptor,centers)
for n=1:size(descriptor,2)
    distance=zeros(1,size(centers,2));
    for m=1:size(centers,2)
        for i=1:size(descriptor,1)
            distance(m)=distance(m)+(descriptor(i,n)-centers(i,m))^2;
        end
    end
    tmp=find(distance==min(distance));
    assignments(n)=tmp(1);
end