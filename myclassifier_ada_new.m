function resultStr = myclassifier_ada_new(imgPath)
% MYCLASSIFIER_ADA_NEW - Predict captcha digits using pre-trained SVM
% INPUT: imgPath - path to a captcha image
% OUTPUT: resultStr - formatted string "filename, d1,d2,d3,d4"

persistent svmModel

% Load SVM model once
if isempty(svmModel)
    fprintf('Loading trained SVM model...\n');
    load('svmModel.mat','svmModel');
end

% Extract filename (e.g., '0002')
[~, name, ~] = fileparts(imgPath);
filename = regexp(name,'\d+','match','once');

% Preprocess & segment digits
digitImages = preprocess_and_segment(imgPath);

% Predict digits
digits = zeros(1,4);  % always 4 digits
for i = 1:4
    feat = extractHOGFeatures(digitImages{i}, 'CellSize',[4 4],'BlockSize',[2 2]);
    digits(i) = str2double(string(predict(svmModel, feat)));
end

% Format output string
resultStr = sprintf('%s, %d,%d,%d,%d', filename, digits);

end

%% ----------------- Helper Functions -----------------
function digitImages = preprocess_and_segment(imgPath)
% Preprocess captcha and segment into individual digit images (1x4 cell)

I = imread(imgPath);

% --- RGB fusion ---
R = I(:,:,1); G = I(:,:,2);
RG_gray = mat2gray(0.5*double(R) + 0.5*double(G));

% --- FFT donut notch filter ---
F = fftshift(fft2(RG_gray));
[h,w] = size(F); cx = floor(w/2)+1; cy = floor(h/2)+1;
inner_r = 6; outer_r = 9;
[x,y] = meshgrid(1:w,1:h);
dist = sqrt((x-cx).^2 + (y-cy).^2);
mask = ~(dist >= inner_r & dist <= outer_r);
F_clean = ifftshift(F .* mask);
I_clean = mat2gray(real(ifft2(F_clean)));

% --- Thresholding & Morphology ---
BW = imbinarize(I_clean, graythresh(I_clean)); BW = ~BW;
BW = imerode(BW, strel('disk',4));
BW = bwareaopen(BW,100);
BW = imdilate(BW, strel('disk',1));

% --- Connected components ---
cc = bwconncomp(BW,8); 
stats = regionprops(cc,'BoundingBox','Area');
areas = [stats.Area]; 
valid = find(areas > 260 & areas < numel(BW)*0.06);

if isempty(valid)
    digitImages = repmat({zeros(28,28)},1,4);
    return
end

boxes = vertcat(stats(valid).BoundingBox);
[~, order] = sort(boxes(:,1));
boxes = boxes(order,:);

% --- Crop digits & pad to 4 ---
nBoxes = size(boxes,1);
digitImages = repmat({zeros(28,28)},1,4); % always 4
for i = 1:min(nBoxes,4)
    digitImages{i} = imresize(imcrop(BW, boxes(i,:)), [28 28]);
end

end
