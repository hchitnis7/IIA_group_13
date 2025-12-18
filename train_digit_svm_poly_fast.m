clc; clear;

rng(42);  % reproducibility

%% -------------------------------------------------
% Paths
%% -------------------------------------------------
dataDir   = 'Train/';
labelFile = fullfile(dataDir,'labels.txt');

%% -------------------------------------------------
% Read labels.txt
%% -------------------------------------------------
fid = fopen(labelFile,'r');
C = textscan(fid,'%s %f %f %f %f','Delimiter',',');
fclose(fid);

fileIDs = C{1};
Yall    = [C{2}, C{3}, C{4}, C{5}];

N      = numel(fileIDs);
nSlots = 4;

%% -------------------------------------------------
% First pass: determine feature length
%% -------------------------------------------------
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

fprintf('Feature length = %d\n', featLen);

%% -------------------------------------------------
% Allocate storage
%% -------------------------------------------------
X = zeros(N, nSlots, featLen, 'single');   % single saves memory
Y = zeros(N, nSlots, 'uint8');

%% -------------------------------------------------
% Feature extraction (NORMAL FOR LOOP)
%% -------------------------------------------------
fprintf('Extracting features...\n');

for i = 1:N
    imgPath = fullfile(dataDir, sprintf('captcha_%s.png', fileIDs{i}));
    I = imread(imgPath);

    [X4, ~, info] = Full_FeatureExtraction(I);

    if isfield(info,'failed') && info.failed
        X(i,:,:) = 0;
    else
        X(i,:,:) = single(X4);
    end

    Y(i,:) = uint8(Yall(i,:));
end

%% -------------------------------------------------
% Train SVMs (polynomial kernel)
%% -------------------------------------------------
fprintf('\nTraining polynomial SVMs...\n');

svmModels   = cell(1,nSlots);
bestParams  = cell(1,nSlots);

for s = 1:nSlots
    fprintf('\n--- Slot %d ---\n', s);

    Xs = squeeze(X(:,s,:));
    Ys = Y(:,s);

    %% ---- Subsample for tuning (40%)
    idx = randperm(size(Xs,1), round(0.4 * size(Xs,1)));
    Xs_sub = Xs(idx,:);
    Ys_sub = Ys(idx);

    %% ---- Base learner (POLY, DEGREE FIXED = 2)
    t = templateSVM( ...
        'KernelFunction','polynomial', ...
        'PolynomialOrder', 2, ...
        'Standardize', true);

    %% ---- Bayesian optimization (FAST)
    svm_tuned = fitcecoc( ...
        Xs_sub, Ys_sub, ...
        'Learners', t, ...
        'Coding','onevsall', ...
        'ClassNames',0:9, ...
        'OptimizeHyperparameters',{'BoxConstraint','KernelScale'}, ...
        'HyperparameterOptimizationOptions',struct( ...
            'KFold',3, ...
            'MaxObjectiveEvaluations',15, ...
            'UseParallel',false, ... % Disabled parallel
            'ShowPlots',false, ...
            'Verbose',0));

    %% ---- Extract best hyperparameters
    bp = svm_tuned.HyperparameterOptimizationResults.XAtMinObjective;
    bestParams{s} = bp;

    fprintf('Best BoxConstraint = %.3f\n', bp.BoxConstraint);
    fprintf('Best KernelScale  = %.3f\n', bp.KernelScale);

    %% ---- LOCKED final training (FAST + STABLE)
    t_locked = templateSVM( ...
        'KernelFunction','polynomial', ...
        'PolynomialOrder', 2, ...
        'KernelScale', bp.KernelScale, ...
        'BoxConstraint', bp.BoxConstraint, ...
        'Standardize', true);

    svmModels{s} = fitcecoc( ...
        Xs, Ys, ...
        'Learners', t_locked, ...
        'Coding','onevsall', ...
        'ClassNames',0:9);
end

%% -------------------------------------------------
% Save model
%% -------------------------------------------------
save('digit_svm_poly_fast.mat','svmModels','bestParams','-v7.3');

fprintf('\nTraining complete. Model saved as digit_svm_poly_fast.mat\n');
