function topfeatures = rankingfeat(TrainMat, LabelTrain)
%% input: TrainMat - a NxM matrix that contains the full list of features
%% of training data. N is the number of training samples and M is the
%% dimension of the feature. So each row of this matrix is the face
%% features of a single person.
%%        LabelTrain - a Nx1 vector of the class labels of training data

%% output: topfeatures - a Kx2 matrix that contains the information of the
%% top 1% features of the highest variance ratio. K is the number of
%% selected feature (K = ceil(M*0.01)). The first column of this matrix is
%% the index of the selected features in the original feature list. So the
%% range of topfeatures(:,1) is between 1 and M. The second column of this
%% matrix is the variance ratio of the selected features.


N=size(TrainMat,1);  M=size(TrainMat,2);

VAR=var(TrainMat);
%VAR is a row vector (1xM) containing the variance of each column(feature)
%for all classes in TrainMat £¨Numerator£©
class=unique(LabelTrain);
VARk=zeros(length(class),M);
%VARk is a matrix that each row contains variance of features belongs to the
%class. the number of rows in VARk equals the number of classes
for i=1:length(class)
    row_index = LabelTrain(:,1)==class(i);
    VARk(i,:) = var(TrainMat(row_index,:));
end

VR=zeros(1,M);
SUM=sum(VARk);
for j=1:M
    VR(j)=length(class)*VAR(j)/SUM(j);
end
%VR is a 1xM matrix storing the variance ratio of M features
VRdescend=sort(VR,'descend');
VRdescend=VRdescend(~isnan(VRdescend));

K = ceil(M*0.01);
topfeatures(2,:) = VRdescend(1:K);
for i=1:K
    topfeatures(1,i) = find(VR==topfeatures(2,i));
end
%choose the top 1% features based on VR and put in topfeatures
topfeatures=topfeatures';

