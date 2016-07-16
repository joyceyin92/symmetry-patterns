clear all;
load('F:\IOS\EEG - Confidential\allfeatures.mat');
clear feature_names;

num_class=16;
data=features;
clear features;
for i=1:num_class
    label=zeros(348,1);
    label(:,1)=i;
    data{i}=[label data{i}];
end
AllMat=cell2mat(data);
countfeat(size(AllMat,2)-1,1)=0;
clear data;
%%
for i=1:30
    [TrainMat, LabelTrain, TestMat, LabelTest]= randomDivideMulti(AllMat);
    topfeatures = rankingfeatAVR(TrainMat, LabelTrain);
    countfeat(topfeatures(:,1),1) = countfeat(topfeatures(:,1),1)+1;
    i
end
countdescend=sort(countfeat,'descend');
for i=1:3
    temp=find(countdescend(i)==countfeat);
    coord(i)=temp(1);
end
%%
%coord=[49221,49378,48597];
c=colormap(jet(num_class)); 
for j=1:num_class
    idx= LabelTrain(:,1)==j;
    PLOT=TrainMat(idx,coord);
    for m=1:size(PLOT,1)
        grid on;
        plot3(PLOT(m,1),PLOT(m,2),PLOT(m,3),'o','color',c(j,:),'MarkerSize',3);
        hold on;
    end
end
hold off;
