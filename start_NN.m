clear all;
load('E:\IOS\EEG - Confidential\allfeatures.mat');
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
for run=1:1
    [TrainMat, LabelTrain, TestMat, LabelTest]= randomDivideMulti(AllMat);
    
    %%
    topfeatures = rankingfeatAVR(TrainMat, LabelTrain);
    
    %forwardselected = forwardselection(TrainMat, LabelTrain, topfeatures);
    
    %% Setup the parameters
    input_layer_size  = size(topfeatures,1);
    hidden_layer_size = 700;%2*input_layer_size;   %  hidden units
    
    %% Implement Regularization
    X=TrainMat(:,topfeatures(:,1)); y=LabelTrain;
    
    lambda = 1;
    
    %J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
    %                  num_class, X, y, lambda);
    
    %%  Sigmoid Gradient
    g = sigmoidGradient([1 -0.5 0 0.5 1]);
    
    %% Initializing Pameters
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_class);
    
    % Unroll parameters
    initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
    
    %%  Training NN
    %  change the MaxIter to a larger value to see how more training helps.
    options = optimset('MaxIter', 50);
    
    % Create "short hand" for the cost function to be minimized
    costFunction = @(p) nnCostFunction(p, ...
        input_layer_size, ...
        hidden_layer_size, ...
        num_class, X, y, lambda);
    
    % Now, costFunction is a function that takes in only one argument (the
    % neural network parameters)
    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
    
    % Obtain Theta1 and Theta2 back from nn_params
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
        hidden_layer_size, (input_layer_size + 1));
    
    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
        num_class, (hidden_layer_size + 1));
    %%  Implement Predict
    
    pred = predict(Theta1, Theta2, X);
    
    AccuracyTrain(run,1)=mean(double(pred == y)) * 100
    
    %%
    %%%test set
    X=TestMat(:,topfeatures(:,1));
    predTest = predict(Theta1, Theta2, X);
    AccuracyTest(run,1)=mean(double(predTest == LabelTest)) * 100;
end
avgTrain=mean(AccuracyTrain);
avgTest=mean(AccuracyTest);