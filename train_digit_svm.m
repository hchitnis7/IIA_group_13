clc; clear;

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
nSlots = 4;   % fixed by your pipeline

% -------------------------------------------------
% First pass: determine feature length
% -------------------------------------------------
fprintf('Determining feature length...\n');

for i = 1:N
    imgPath = fullfile(dataDir, sprintf('captcha_%s.png', fileIDs{i}));
    I = imread(imgPath);
    [X4, ~, info] = Full_FeatureExtraction(I);

    if ~isfield(info,'failed') || ~info.failed
        featLen = size(X4,2);
        break;
    end
end

if ~exist('featLen','var')
    error('All samples failed feature extraction.');
end

fprintf('Feature length = %d\n', featLen);

% -------------------------------------------------
% Allocate storage
% -------------------------------------------------
X = zeros(N, nSlots, featLen);
Y = zeros(N, nSlots);

% -------------------------------------------------
% Feature extraction loop
% -------------------------------------------------
fprintf('Extracting features...\n');

for i = 1:N
    imgPath = fullfile(dataDir, sprintf('captcha_%s.png', fileIDs{i}));
    I = imread(imgPath);

    [X4, ~, info] = Full_FeatureExtraction(I);

    if isfield(info,'failed') && info.failed
        X(i,:,:) = 0;
    else
        X(i,:,:) = X4;
    end

    Y(i,:) = Yall(i,:);
end

% 
% % -------------------------------------------------
% % Train SVMs (one per slot)
% % -------------------------------------------------
% fprintf('\nTraining SVM classifiers...\n');
% 
% t = templateSVM( ...
%     'KernelFunction','rbf', ...
%     'KernelScale','auto', ...
%     'Standardize',true);
% 
% svmModels = cell(1,nSlots);
% 
% for s = 1:nSlots
%     fprintf('  Training slot %d...\n', s);
% 
%     Xs = squeeze(X(:,s,:));   % NxD
%     Ys = Y(:,s);              % Nx1
% 
%     svmModels{s} = fitcecoc( ...
%         Xs, Ys, ...
%         'Learners', t, ...
%         'Coding','onevsall', ...
%         'ClassNames', 0:9);
% end

% -------------------------------------------------
% Train SVMs (one per slot) — Polynomial Kernel
% -------------------------------------------------
fprintf('\nTraining SVM classifiers (Polynomial kernel)...\n');

svmModels = cell(1,nSlots);

for s = 1:nSlots
    fprintf('  Training slot %d...\n', s);

    Xs = squeeze(X(:,s,:));   % NxD
    Ys = Y(:,s);              % Nx1

    % Remove empty / failed samples
    valid = any(Xs,2);
    Xs = Xs(valid,:);
    Ys = Ys(valid);

    % Fixed SVM template
    t = templateSVM( ...
        'KernelFunction', 'polynomial', ...
        'Standardize', true);

    % Hyperparameter optimization options
    opts = struct( ...
        'Optimizer', 'bayesopt', ...
        'ShowPlots', false, ...
        'Verbose', 1, ...
        'KFold', 5, ...
        'MaxObjectiveEvaluations', 15);

    svmModels{s} = fitcecoc( ...
        Xs, Ys, ...
        'Learners', t, ...
        'Coding', 'onevsall', ...
        'ClassNames', 0:9, ...
        'OptimizeHyperparameters', { ...
            'BoxConstraint', ...
            'KernelScale', ...
            'PolynomialOrder'}, ...
        'HyperparameterOptimizationOptions', opts);
end



% -------------------------------------------------
% Save trained model
% -------------------------------------------------
save('digit_svm_model_poly.mat','svmModels');

fprintf('\nTraining complete. Model saved \n');
