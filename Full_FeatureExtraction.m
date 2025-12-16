function [X4, patches4, info] = Full_FeatureExtraction(I, params)
% FULL_FEATUREEXTRACTION
%   Full feature extraction pipeline:
%   - Calls preprocessCaptcha
%   - Performs equal-width digit slicing
%   - HOG feature extraction
%
% Outputs:
%   X4       : 4 x D HOG feature matrix (D fixed)
%   patches4 : 1 x 4 cell array of digit binary patches
%   info     : struct with diagnostic info

    if nargin < 2 || isempty(params)
        params = default_feat_params();
    end


    % ------------Preprocess------------

    [BWfull, ~, info] = preprocessCaptcha(I);

    % Early exit
    if isfield(info,'failed') && info.failed
        X4 = zeros(params.maxSlots, get_featLen(params));
        patches4 = cell(1, params.maxSlots);
        return;
    end


    % -------------Bounding box + nDigits------------

    [yIdx, xIdx] = find(BWfull);

    if isempty(xIdx)
        info.failed = true;
        X4 = zeros(params.maxSlots, get_featLen(params));
        patches4 = cell(1, params.maxSlots);
        return;
    end

    y1 = min(yIdx); y2 = max(yIdx);
    x1 = min(xIdx); x2 = max(xIdx);

    bboxW = x2 - x1 + 1;
    bboxH = y2 - y1 + 1;

    isFour = bboxW > params.BBOXW_THRESH;
    nDigits = params.minDigits + isFour;

    info.bboxW   = bboxW;
    info.bboxH   = bboxH;
    info.isFour  = isFour;
    info.nDigits = nDigits;

   
    BWseg = BWfull(y1:y2, x1:x2);
    BWseg = bwareaopen(BWseg, params.cropAreaOpen);

    % ------------Equal-width slicing---------------
    Wseg = size(BWseg,2);
    edges = round(linspace(1, Wseg+1, nDigits+1));
    digitPatches = cell(1,nDigits);

    for k = 1:nDigits
        xL = edges(k); xR = edges(k+1)-1;
        digitPatches{k} = trimBinary(BWseg(:, xL:xR));
    end


    % ------------Map into fixed slots------------

    patches4 = cell(1, params.maxSlots);

    if isFour
        patches4(1:params.maxSlots) = digitPatches;
    else
        if params.forceLeading0
            patches4{1} = [];  % leading zero for 3 digit captchas
            patches4(2:4) = digitPatches;
        else
            patches4(1:3) = digitPatches;
        end
    end


    % ------------HOG extraction------------

    featLen = get_featLen(params);
    X4 = zeros(params.maxSlots, featLen);

    for s = 1:params.maxSlots
        patch = patches4{s};
        if isempty(patch) || ~any(patch(:))
            continue;
        end
        f = hog_from_patch(patch, params);
        % Pad / truncate for fixed length
        if numel(f) < featLen
            f(end+1:featLen) = 0;
        elseif numel(f) > featLen
            f = f(1:featLen);
        end
        X4(s,:) = f;
    end

end


%% ------------DEFAULT FEATURE PARAMETERS------------

function params = default_feat_params()
    params = struct();

    % HOG features
    params.hogSize     = [28 28];
    params.hogCellSize = [7 7];

    % Digit slicing / slot mapping
    params.maxSlots      = 4;
    params.minDigits     = 3;
    params.forceLeading0 = true;

    % Bounding box width threshold
    params.BBOXW_THRESH  = 240;
    params.cropAreaOpen  = 30;
end


%% ------------HOG Extraction Helper------------

function x = hog_from_patch(patch, params)
    patch = logical(patch);
    patch = trimBinary(patch);
    if isempty(patch) || ~any(patch(:))
        dummy = false(params.hogSize);
        x = extractHOGFeatures(dummy, 'CellSize', params.hogCellSize);
        x(:) = 0;
        return;
    end
    im = imresize(patch, params.hogSize, 'nearest');
    x = extractHOGFeatures(im, 'CellSize', params.hogCellSize);
end


%% ------------Feature length helper------------

function featLen = get_featLen(params)
    dummy = false(params.hogSize);
    featLen = numel(extractHOGFeatures(dummy,'CellSize',params.hogCellSize));
end


%% ------------Utility: Trim binary patch------------

function B = trimBinary(B)
    if isempty(B) || ~any(B(:)), return; end
    rows = any(B,2); cols = any(B,1);
    B = B(rows,cols);
end
