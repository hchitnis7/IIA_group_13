function S = myclassifier(im)
% MYCLASSIFIER
%   Classify a CAPTCHA image into a 1x4 digit vector.
%   Uses trained SVM models and Full_FeatureExtraction.
%
% Output format:
%   - Always 1x4
%   - Three-digit CAPTCHA is padded with leading zero

    % -------------------------------------------------
    % Persistent model loading (loaded once)
    % -------------------------------------------------
    persistent svmModels isLoaded

    if isempty(isLoaded)
        modelData = load('digit_svm_model.mat','svmModels');
        svmModels = modelData.svmModels;
        isLoaded = true;
    end

    % -------------------------------------------------
    % Feature extraction
    % -------------------------------------------------
    [X4, ~, info] = Full_FeatureExtraction(im);

    % Safety fallback
    if isfield(info,'failed') && info.failed
        S = [0 0 0 0];
        return;
    end

    % -------------------------------------------------
    % Slot-wise prediction
    % -------------------------------------------------
    S = zeros(1,4);

    for s = 1:4
        S(s) = predict(svmModels{s}, X4(s,:));
    end

    % -------------------------------------------------
    % Enforce leading-zero convention
    % -------------------------------------------------
    % If it is a 3-digit CAPTCHA, slot-1 must be zero
    if info.nDigits == 3
        S(1) = 0;
    end
end
