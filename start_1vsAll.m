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
%AllMat is a matrix contains all data and features. each row is a data
%point and the first column is label.

clear data;
%%
MisClass=zeros(num_class,num_class);
for run=1:10
    [TrainMat, LabelTrain, TestMat, LabelTest]= randomDivideMulti(AllMat);
    
    topfeatures = rankingfeatAVR(TrainMat, LabelTrain);
    
    %%
    % start classification
    X=TrainMat(:,topfeatures(:,1)); 
    y=LabelTrain;
    
    lambda = 0.1;
    [all_theta] = oneVsAll(X, y, num_class, lambda);
    
    pred = predictOneVsAll(all_theta, X);
    AccuracyTrain(run,1)=mean(double(pred == y)) * 100;
    %%
    %%%test set
    X=TestMat(:,topfeatures(:,1));
    predTest = predictOneVsAll(all_theta, X);
    AccuracyTest(run,1)=mean(double(predTest == LabelTest)) * 100;
    
    %%
    for i = 1:num_class
        row_index = LabelTest == i;
        Test{i} = TestMat(row_index,topfeatures(:,1));
        predTest = predictOneVsAll(all_theta, Test{i});
        for j=1:num_class
            MisClass(i,j)=MisClass(i,j)+length(find(predTest==j));
            %MisClass is a 16x16 matrix analysis mis-classify numbers in test
            %set. Each row represents a whole class of data.
            %MisClass(i,j) represents the number of data that bolongs to class
            %i was classified into class j
        end
        EachClass(i,run)=mean(double(predTest == i)) * 100;
        %EachClass record the classification rate of each classes. (16xrun_num)
    end
    run
end
%%
avgEach=mean(EachClass,2);
avgTrain=mean(AccuracyTrain);
avgTest=mean(AccuracyTest);
MisClass=MisClass/run;