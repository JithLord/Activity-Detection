%% Loading Training Data

clc; clear all; clc
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
D = pdist(activity_trainX{:,:},"cityblock");
[Y,eigen] = cmdscale(D); 
% Used to find the configuration Matrix and it's corresponding Eigen values
figure % Eigen values
pareto(eigen)
size_tab = size(activity_trainX,2);  %= 20

%% PCA Principal Component Analysis
[pcs,scrs,~,~,pexp] = pca(activity_trainX{:,:});
pareto(pexp)

%% Fitting Models
model_knn   = fitcknn(activity_trainX,activity_trainY,"NumNeighbors",2);
model_tree  = fitctree(activity_trainX,activity_trainY);
model_nb    = fitcnb(activity_trainX,activity_trainY);
model_ecoc  = fitcecoc(activity_trainX,activity_trainY);
model_da    = fitcdiscr(activity_trainX,activity_trainY);

%% Predicting for training data
pred_knn1    = predict(model_knn, activity_trainX);
pred_tree1   = predict(model_tree, activity_trainX);
pred_nb1     = predict(model_nb, activity_trainX);
pred_ecoc1    = predict(model_ecoc, activity_trainX);
pred_da1     = predict(model_da, activity_trainX);

%% Predicting for testing data
pred_knn    = predict(model_knn, activity_testX);
pred_tree   = predict(model_tree, activity_testX);
pred_nb     = predict(model_nb, activity_testX);
pred_ecoc    = predict(model_ecoc, activity_testX);
pred_da     = predict(model_da, activity_testX);

%% Loss for testing and training data
final_pred_train = [mode([pred_knn1'; pred_tree1'; pred_nb1'; pred_ecoc1';pred_da1'])]';
final_pred_test  = [mode([pred_knn'; pred_tree'; pred_nb'; pred_ecoc';pred_da'])]';

final_acc_train = sum([final_pred_train == activity_trainY])./numel([final_pred_train == activity_trainY]);
final_acc_test = sum([final_pred_test == activity_testY])./numel([final_pred_test == activity_testY]);

disp("The loss for training data is "+string(final_acc_train))
disp("The loss for testing data is "+string(final_acc_test))
