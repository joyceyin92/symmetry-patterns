function [classify]=leastsquare(class,train,labeltrain,test)
X=train;
X=[ones(size(train,1),1) train];
T=zeros(size(train,1),class);
for i=1:size(train,1)
    T(i,labeltrain(i,1)+1)=1;
end
W=(X'*X)\(X'*T);

x=test';
x=[ones(1,size(test,1));x];
classify=zeros(size(x,2),1);
for t=1:size(x,2)
    y=W'*x(:,t);
    [~,I]=max(y);
    classify(t)=I;
end