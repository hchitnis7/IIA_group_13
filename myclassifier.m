function S = myclassifier(im)
% MYCLASSIFIER
%   Classify a CAPTCHA image into a 1x4 digit vector.
%   Uses trained SVM models and Full_FeatureExtraction.
%
% Output format:
%   - Always 1x4
%   - Three-digit CAPTCHA is padded with leading zero

    % model loading (loaded once)
    persistent svmModels classNames isLoaded

    if isempty(isLoaded)
        modelData = load('digit_svm_model_0345_ppup.mat','svmModels');
        svmModels = modelData.svmModels;
        classNames = svmModels{1}.ClassNames;   % e.g. [0 3 4 5]
        isLoaded = true;
    end

    % Feature extraction
    [X4, ~, info] = Full_FeatureExtraction(im);

    % Safety fallback
    if isfield(info,'failed') && info.failed
        S = [0 0 0 0];
        return;
    end

    % Slot-wise prediction + scores
    S = zeros(1,4);
    scores = cell(1,4);

    for s = 1:4
        [S(s), scores{s}] = predict(svmModels{s}, X4(s,:));
    end

    % -------------------------------------------------
    % Enforce structural constraint using model scores
    % 0 allowed ONLY in slot 1
    % -------------------------------------------------
    for s = 2:4
        if S(s) == 0
            sc = scores{s};
            sc(classNames == 0) = -Inf;   % forbid 0
            [~, idx] = max(sc);
            S(s) = classNames(idx);
        end
    end

    % -------------------------------------------------
    % Enforce leading-zero convention for 3-digit CAPTCHA
    % -------------------------------------------------
    if info.nDigits == 3
        S(1) = 0;
    end
end
