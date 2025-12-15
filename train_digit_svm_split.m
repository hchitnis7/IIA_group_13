clc; clear;

rng(42);   % for reproducibility

% -------------------------------------------------
% Paths
% -------------------------------------------------
dataDir   = 'Train/';
labelFile = fullfile(dataDir,'labels.txt');

% -------------------------------------------------
% Read labels.txt
% -------------------------------------------------
fid = fopen(labelFile,'r');
C = textscan(fid,'%s %f %f %f %f','Delimiter',',');
fclose(fid);

fileIDs = C{1};
Yall    = [C{2}, C{3}, C{4}, C{5}];   % Nx4 labels

N      = numel(fileIDs);
nSlots = 4;

% -------------------------------------------------
% Train / Test split (80 / 20)
% -------------------------------------------------
idx = randperm(N);

nTrain = round(0.8 * N);
trainIdx = idx(1:nTrain);
testIdx  = idx(nTrain+1:end);

fileIDs_train = fileIDs(trainIdx);
Y_train       = Yall(trainIdx,:);

fileIDs_test  = fileIDs(testIdx);
Y_test        = Yall(testIdx,:);

fprintf('Train samples: %d\n', numel(trainIdx));
fprintf('Test samples : %d\n', numel(testIdx));

% -------------------------------------------------
% Determine feature length (from training set)
% -------------------------------------------------
fprintf('Determining feature length...\n');

for i = 1:numel(trainIdx)
    imgPath = fullfile(dataDir, ...
        sprintf('captcha_%s.png', fileIDs_train{i}));

    I = imread(imgPath);
    [X4, ~, info] = Full_FeatureExtraction(I);

    if ~isfield(info,'failed') || ~info.failed
        featLen = size(X4,2);
        break;
    end
end

if ~exist('featLen','var')
    error('All training samples failed feature extraction.');
end

fprintf('Feature length = %d\n', featLen);

% -------------------------------------------------
% Allocate TRAINING feature storage
% -------------------------------------------------
Xtrain = zeros(numel(trainIdx), nSlots, featLen);

% -------------------------------------------------
% Extract TRAINING features
% -------------------------------------------------
fprintf('Extracting training features...\n');

for i = 1:numel(trainIdx)
    imgPath = fullfile(dataDir, ...
        sprintf('captcha_%s.png', fileIDs_train{i}));

    I = imread(imgPath);
    [X4, ~, info] = Full_FeatureExtraction(I);

    if isfield(info,'failed') && info.failed
        Xtrain(i,:,:) = 0;
    else
        Xtrain(i,:,:) = X4;
    end
end

% -------------------------------------------------
% Train SVMs (one per slot)
% -------------------------------------------------
fprintf('\nTraining SVM classifiers...\n');

t = templateSVM( ...
    'KernelFunction','rbf', ...
    'KernelScale','auto', ...
    'Standardize',true);

svmModels = cell(1,nSlots);

for s = 1:nSlots
    fprintf('  Training slot %d...\n', s);

    Xs = squeeze(Xtrain(:,s,:));   % NxD
    Ys = Y_train(:,s);             % Nx1

    svmModels{s} = fitcecoc( ...
        Xs, Ys, ...
        'Learners', t, ...
        'Coding','onevsall', ...
        'ClassNames', 0:9);
end

% -------------------------------------------------
% Save trained model
% -------------------------------------------------
save('digit_svm_model.mat','svmModels');

fprintf('\nModel saved to digit_svm_model.mat\n');

% -------------------------------------------------
% Write TEST labels file (IMPORTANT)
% -------------------------------------------------
testLabelFile = fullfile(dataDir,'labels_test.txt');
fid = fopen(testLabelFile,'w');

for i = 1:numel(testIdx)
    fprintf(fid,'%s, %d, %d, %d, %d\n', ...
        fileIDs_test{i}, ...
        Y_test(i,1), Y_test(i,2), Y_test(i,3), Y_test(i,4));
end

fclose(fid);

fprintf('Test labels written to %s\n', testLabelFile);
