%% Loading Training Data
clc; 
activity_train = readtable('train\train.csv','PreserveVariableNames',true);
activity_trainX = normalize(activity_train(:,1:end-1));
activity_trainY = categorical(activity_train.Response);
activity_train.Response = activity_trainY;
%% Loading the test data
activity_test = readtable('test\test.csv','PreserveVariableNames',true);
activity_testX = normalize(activity_test(:,1:end-1));
% activity_testY = readtable('test\y_test.csv');
activity_testY = categorical(activity_test.Response);
%% Multi-Dimensional Scaling using Pairwise distances or PDA
close all
D = pdist(activity_trainX{:,:},"cityblock");
[Y,eigen] = cmdscale(D); 
% Used to find the configuration Matrix and it's corresponding Eigen values
figure % Eigen values
pareto(eigen)
size_tab = size(activity_trainX,2);  %= 20

%% PCA Principal Component Analysis
[pcs,scrs,~,~,pexp] = pca(activity_trainX{:,:});
pareto(pexp)

%% Fitting a Model
model_knn   = fitcknn(activity_trainX,activity_trainY,"NumNeighbors",2);
model_tree  = fitctree(activity_trainX,activity_trainY);
model_nb    = fitcnb(activity_trainX,activity_trainY);
model_ecoc   = fitcecoc(activity_trainX,activity_trainY);


%% Prediction
pred_knn    = predict(model_knn, activity_testX);
pred_tree   = predict(model_tree, activity_testX);
pred_nb     = predict(model_nb, activity_testX);
pred_svm    = predict(model_ecoc, activity_testX);

final_pred  = [mode([pred_knn'; pred_tree'; pred_nb'; pred_svm'])]';
%% Loss 
loss_knn = loss(model_knn, activity_train);
final_loss = sum([final_pred == activity_testY])./numel([final_pred == activity_testY])
