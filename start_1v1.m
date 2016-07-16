clear all;
load('E:\IOS\EEG - Confidential\allfeatures.mat');
clear feature_names;

num_class=16;
data=features;
clear features;
for i=1:num_class
    y=zeros(348,1);
    y(:,1)=i;
    data{i}=[y data{i}];
end
AllMat=cell2mat(data);
%AllMat is a matrix contains all data and features. each row is a data
%point and the first column is label.

clear data;
%%
for run=1:1
    [TrainMat, LabelTrain, TestMat, LabelTest]= randomDivideMulti(AllMat);
    
    topfeatures = rankingfeatAVR(TrainMat, LabelTrain);
        
    %%%group data into pairs
    combine=nchoosek(1:16,2);
    voteTrain=zeros(size(TrainMat,1),num_class);
    voteTest=zeros(size(TestMat,1),num_class);
    
    for n=1:size(combine,1)
        X=[];
        y=[];
        row_index_1 = LabelTrain(:,1)==combine(n,1);
        row_index_2 = LabelTrain(:,1)==combine(n,2);
        X=[X;TrainMat(row_index_1,:);TrainMat(row_index_2,:)];
        y=[y;LabelTrain(row_index_1,:);LabelTrain(row_index_2,:)];
        
        %%%%Gaussian Kernel
        X=X(:,topfeatures(:,1));
        
        temp=unique(y);
        %change label of chosen pairs into 1/0
        y(y==temp(1))=0;
        y(y==temp(2))=1;
        
        x1 = [1 2 1]; x2 = [0 4 -1]; C=1; sigma=0.1;
        %[C, sigma] = dataset3Params(X, y, Xval, yval);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        %model = svmTrain(X, y, C, @linearKernel, 1e-3, 20); %linear kernel
        
        %%%Train set
        pred = svmPredict(model, TrainMat(:,topfeatures(:,1)));
        
       %change label back
        for j=1:size(pred,1)
            if pred(j)==0
                pred(j)=temp(1);
            else if pred(j)==1
                    pred(j)=temp(2);
                end
            end
        end
        
        for i=1:size(TrainMat)
            voteTrain(i,pred(i)) = voteTrain(i,pred(i))+1;
        end
        
 %%  
        %%Test set
        pred = svmPredict(model, TestMat(:,topfeatures(:,1)));
        
        for j=1:size(pred,1)
            if pred(j)==0
                pred(j)=temp(1);
            else if pred(j)==1
                    pred(j)=temp(2);
                end
            end
        end
        
        for i=1:size(TestMat)
            voteTest(i,pred(i)) = voteTest(i,pred(i))+1;
        end
    n 
    end
    %Train accuracy
     [~,I]=max(voteTrain,[],2);
     for z=1:num_class
         correctTrain(z,run)=length( I((z-1)*174+1:z*174,1)==z ); 
         %correctTrain is a 16*run matrix store the number of correct label in each class during each run
     end
         
        accuracyTrain(run,1)=0;
        for i=1:size(TrainMat,1)
            if I(i,1)==LabelTrain(i)
                accuracyTrain(run,1)=accuracyTrain(run,1)+1;
            end
        end
        accuracyTrain(run,1)=accuracyTrain(run,1)/size(LabelTrain,1);
        
       %Test accuracy
    [~,I]=max(voteTest,[],2);
     for z=1:num_class
         correctTest(z,run)=length( I((z-1)*174+1:z*174,1)==z );
        end
    accuracyTest(run,1)=0;
    for i=1:size(TestMat,1)
        if I(i,1)==LabelTest(i)
            accuracyTest(run,1)=accuracyTest(run,1)+1;
        end
    end
    
    accuracyTest(run,1)=accuracyTest(run,1)/size(LabelTest,1);
end
avgTrain=mean(accuracyTrain);
avgTest=mean(accuracyTest);

