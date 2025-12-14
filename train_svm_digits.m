%% train_svm_digits.m
% Train SVM for digit recognition and save the model

clear; clc;

trainFolder = 'C:\Users\Harsh Chitnis\Desktop\IIA_group_13\Train';
labelsFile = fullfile(trainFolder,'labels.txt');

%% --- Read labels ---
fid = fopen(labelsFile,'r');
data = textscan(fid, '%s %d %d %d %d','Delimiter',',');
fclose(fid);

imgIDs = data{1};
labelsMat = [data{2}, data{3}, data{4}, data{5}];  % 4 digits per image

%% --- Load and process images ---
trainImages = {};  % cell array for all digit images
trainLabels = [];  % labels vector

fprintf('Processing %d images...\n', numel(imgIDs));
for k = 1:numel(imgIDs)
    imgPath = fullfile(trainFolder,sprintf('captcha_%s.png', imgIDs{k}));
    I = imread(imgPath);
    
    % Segment digits
    digitImgs = preprocess_and_segment(I);
    nDigitsFound = numel(digitImgs);
    
    % Initialize padded digit images with blanks
    digitImgsPadded = repmat({zeros(28,28)}, 1, 4);
    
    % Place detected digits at rightmost positions safely
    for j = 1:nDigitsFound
        idx = 4 - nDigitsFound + j;    % position in padded array
        if idx >= 1 && idx <= 4
            digitImgsPadded{idx} = digitImgs{j};
        end
    end
    
    % Read the corresponding labels for this captcha
    labelsRow = labelsMat(k,:);       % original labels
    labelsRow(labelsRow==0) = [];      % remove existing padding zeros
    nLabels = numel(labelsRow);
    
    % Initialize padded labels with zeros
    labelsPadded = zeros(1,4);
    
    % Place actual labels at rightmost positions safely
    for j = 1:nLabels
        idx = 4 - nLabels + j;
        if idx >= 1 && idx <= 4
            labelsPadded(idx) = labelsRow(j);
        end
    end
    
    % Add to training arrays
    for j = 1:4
        trainImages{end+1} = digitImgsPadded{j};
        trainLabels(end+1,1) = labelsPadded(j);
    end
    
    % Progress printing
    if mod(k,50) == 0
        fprintf('Processed %d/%d images\n', k, numel(imgIDs));
    end
end

trainLabels = categorical(trainLabels);  % SVM expects categorical

%% --- Extract features ---
fprintf('Extracting HOG features...\n');
feats = [];
for i = 1:numel(trainImages)
    feats(i,:) = extractHOGFeatures(trainImages{i}, 'CellSize',[4 4],'BlockSize',[2 2]);
end

%% --- Train SVM ---
fprintf('Training SVM...\n');
svmModel = fitcecoc(feats, trainLabels, 'Learners','linear','Coding','onevsall');
fprintf('Training complete!\n');

%% --- Save model ---
save('svmModel.mat','svmModel');
fprintf('Model saved to svmModel.mat\n');

%% ---------------- Helper: Preprocessing ----------------
function digitImages = preprocess_and_segment(I)
    % --- RGB fusion ---
    R = I(:,:,1); G = I(:,:,2);
    RG_gray = mat2gray(0.5*double(R)+0.5*double(G));

    % --- FFT donut notch ---
    F = fftshift(fft2(RG_gray));
    [h,w] = size(F); cx=floor(w/2)+1; cy=floor(h/2)+1;
    inner_r=6; outer_r=9;
    [x,y] = meshgrid(1:w,1:h);
    dist = sqrt((x-cx).^2 + (y-cy).^2);
    mask = ~(dist>=inner_r & dist<=outer_r);
    F_clean = ifftshift(F.*mask);
    I_clean = mat2gray(real(ifft2(F_clean)));

    % --- Binarization + Morphology ---
    BW = imbinarize(I_clean, graythresh(I_clean)); BW = ~BW;
    BW = imerode(BW, strel('disk',4));
    BW = bwareaopen(BW,100);
    BW = imdilate(BW, strel('disk',1));

    % --- Connected components ---
    cc = bwconncomp(BW,8); 
    stats = regionprops(cc,'BoundingBox','Area');
    areas = [stats.Area]; 
    valid = find(areas>260 & areas<numel(BW)*0.06);
    
    if isempty(valid)
        digitImages = {};  % no digits found
        return
    end

    boxes = vertcat(stats(valid).BoundingBox);

    % Sort digits left to right
    [~, order] = sort(boxes(:,1));
    boxes = boxes(order,:);

    % Crop digits
    nBoxes = size(boxes,1);
    digitImages = cell(nBoxes,1);
    for i = 1:nBoxes
        digitImages{i} = imresize(imcrop(BW, boxes(i,:)), [28 28]);
    end
end
