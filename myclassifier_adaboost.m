function S = myclassifier_adaboost(im)
% MYCLASSIFIER_ADABOOST_VERBOSE - Predict captcha digits using multi-class AdaBoost
% Input: im - RGB image
% Output: S - 1x4 vector of digits

    persistent model
    trainFolder = 'C:/Users/Harsh Chitnis/Desktop/IIA_group_13/Train';

    if isempty(model)
        fprintf('--- Training AdaBoost model ---\n');

        % --- Load labels ---
        fprintf('Loading labels...\n');
        fid = fopen(fullfile(trainFolder,'labels.txt'));
        if fid == -1
            error('Cannot open labels.txt');
        end
        data = textscan(fid, '%s %d %d %d %d', 'Delimiter', ',');
        fclose(fid);
        labels = [data{2}, data{3}, data{4}, data{5}];
        fprintf('Loaded %d labels.\n', size(labels,1));

        % --- Load training images ---
        N = numel(data{1});
        feats = [];
        labelsVec = [];
        fprintf('Loading and processing images...\n');
        for k = 1:N
            imgPath = fullfile(trainFolder, sprintf('captcha_%04d.png', k));
            if ~isfile(imgPath)
                warning('Image not found: %s', imgPath);
                continue
            end
            I = imread(imgPath);
            digitImages = preprocess_and_segment_verbose(I);
            for j = 1:4
                feat = extract_features_digit(digitImages{j});
                feats(end+1,:) = feat;
                labelsVec(end+1,1) = labels(k,j);
            end
            if mod(k,20)==0
                fprintf('Processed %d/%d images...\n', k, N);
            end
        end

        fprintf('Training AdaBoost classifier...\n');
        model = fitcensemble(feats, categorical(labelsVec), ...
            'Method','LogitBoost', 'NumLearningCycles',50, 'Learners','Tree');
        fprintf('Training complete!\n');
    end

    % --- Predict ---
    fprintf('Predicting digits for current image...\n');
    digitImages = preprocess_and_segment_verbose(im);
    digits = zeros(1,4);
    for i = 1:4
        digits(i) = str2double(string(predict(model, extract_features_digit(digitImages{i}))));
        fprintf('Digit %d: %d\n', i, digits(i));
    end
    S = digits;
end

%% ---------------- Helper Functions ----------------

function feat = extract_features_digit(digitImg)
    digitImg = logical(digitImg);
    feat = extractHOGFeatures(digitImg,'CellSize',[4 4],'BlockSize',[2 2]);
end

function digitImages = preprocess_and_segment_verbose(I)
    if ischar(I) || isstring(I)
        I = imread(I);
    end

    fprintf('Segmenting image...\n');

    R = I(:,:,1); G = I(:,:,2);
    RG_gray = mat2gray(0.5*double(R)+0.5*double(G));

    % FFT donut notch
    F = fftshift(fft2(RG_gray));
    [h,w] = size(F); cx=floor(w/2)+1; cy=floor(h/2)+1;
    inner_r=6; outer_r=9;
    [x,y] = meshgrid(1:w,1:h);
    dist = sqrt((x-cx).^2 + (y-cy).^2);
    mask = ~(dist>=inner_r & dist<=outer_r);
    F_clean = ifftshift(F.*mask);
    I_clean = mat2gray(real(ifft2(F_clean)));

    % Threshold & morphology
    BW = imbinarize(I_clean, graythresh(I_clean)); BW = ~BW;
    BW = imerode(BW, strel('disk',4));
    BW = bwareaopen(BW,100);
    BW = imdilate(BW, strel('disk',1));

    % Connected components
    cc = bwconncomp(BW,8); stats = regionprops(cc,'BoundingBox','Area');
    areas = [stats.Area]; valid = find(areas>260 & areas<numel(BW)*0.06);

    if isempty(stats) || isempty(valid)
        fprintf('No digits found, returning blank images.\n');
        digitImages = repmat({zeros(28,28)},1,4);
        return
    end

    boxes = vertcat(stats(valid).BoundingBox);
    [~, order] = sort(boxes(:,1));
    boxes = boxes(order,:);
    nBoxes = min(4,size(boxes,1));
    digitImages = cell(1,4);
    for i = 1:4
        if i <= nBoxes
            digitImages{i} = imresize(imcrop(BW, boxes(i,:)), [28 28]);
        else
            digitImages{i} = zeros(28,28);
        end
    end
    fprintf('Found %d digits.\n', nBoxes);
end
