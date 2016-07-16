function forwardselected = forwardselection(TrainMat, LabelTrain, topfeatures)
%% input: TrainMat - a NxM matrix that contains the full list of features
%% of training data. N is the number of training samples and M is the
%% dimension of the feature. So each row of this matrix is the face
%% features of a single person.
%%        LabelTrain - a Nx1 vector of the class labels of training data
%%        topfeatures - a Kx2 matrix that contains the information of the
%% top 1% features of the highest variance ratio. K is the number of
%% selected feature (K = ceil(M*0.01)). The first column of this matrix is
%% the index of the selected features in the original feature list. So the
%% range of topfeatures(:,1) is between 1 and M. The second column of this
%% matrix is the variance ratio of the selected features.

%% output: forwardselected - a Px1 vector that contains the index of the 
%% selected features in the original feature list, where P is the number of
%% selected features. The range of forwardselected is between 1 and M. 
class=length(unique(LabelTrain));
    clear ClassRate;
    clear Classification;
    ClassRate=zeros(size(topfeatures,1),1);
   
    %select the 1st feature
    for in=1:size(topfeatures,1)
        LS=leastsquare(class,TrainMat(:,topfeatures(in,1)),LabelTrain,TrainMat(:,topfeatures(in,1)));
        %LS is a column vector denote the classification result of TrainMat.
        for j=1:size(LS)
            ClassRate(in)=ClassRate(in)+length(find((LS(j)-1)==LabelTrain(j)));
        end
        ClassRate(in)=ClassRate(in)/size(LS,1);
        %ClassRate is a matrix records the Classification Rate for the same
        %round (Same layer in the tree). We need to choose the maximum one.
    end
    nth=find(ClassRate==max(max(ClassRate)));
    nth=nth(1); 
    %if there are multiple same maximum values in Classification rate, what 
    %to do? (Here I choose the first one. Is there any better idea?)
    Classification(1)=ClassRate(nth);
    %Clssification is a column vector storing the maximum classification 
    %rate in the same layer of the tree 
    forwardselected=topfeatures(nth,1);
    OnePercent=topfeatures;
    topfeatures(nth,:)=[];
    %delete the selected feature from topfeatures
    
    
    %select the 2nd feature to the end
    for m=2:size(OnePercent,1)
        ClassRate=zeros(size(topfeatures,1),1);
        for n=1:size(topfeatures,1)
            forwardselected(m,1)=topfeatures(n,1);
            LS=leastsquare(class,TrainMat(:,forwardselected),LabelTrain,TrainMat(:,forwardselected));
            for j=1:size(LS)
                ClassRate(n)=ClassRate(n)+length(find((LS(j)-1)==LabelTrain(j)));
            end
            ClassRate(n)=ClassRate(n)/size(LS,1);
        end
        nth=find(ClassRate==max(max(ClassRate)));
        nth=nth(1); 
        Classification(m,1)=ClassRate(nth,1);
        if (Classification(m,1)<Classification(m-1,1))||Classification(m,1)==1
            break
            %if no additional feature improves the objective evaluation 
            %function or classification rate has achieved 100%, break 
        end
        forwardselected(m,1)=topfeatures(nth,1);
        topfeatures(nth,:)=[];
    end
    
    