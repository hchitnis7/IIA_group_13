function feat = FeatureExtraction(I)
% FEATUREEXTRACTION - preprocesses image and returns a fixed-length feature vector
% INPUT: I - RGB image
% OUTPUT: feat - 1 x F feature vector

% ---------------- PREPROCESS ----------------
R = I(:,:,1); G = I(:,:,2);
RG = mat2gray(0.5*double(R) + 0.5*double(G));
RG = imresize(RG, 0.5);

F = fftshift(fft2(RG));
[h,w] = size(F);
cx = floor(w/2)+1; cy = floor(h/2)+1;
[x,y] = meshgrid(1:w,1:h);
dist = sqrt((x-cx).^2 + (y-cy).^2);
mask = ~(dist>=6 & dist<=9);
Iclean = mat2gray(real(ifft2(ifftshift(F.*mask))));

BW = imbinarize(Iclean,graythresh(Iclean));
BW = ~BW;
BW = imerode(BW,strel('disk',4));
BW = bwareaopen(BW,100);
BW = imdilate(BW,strel('disk',1));

% ---------------- Feature Extraction ----------------
% Instead of cropping, resize the processed image to fixed size and flatten
fixed_size = [28 80]; % height x width (adjustable)
BW_resized = imresize(BW, fixed_size);
feat = FeatureExtraction(I);
feat = double(feat); % ensure numeric
end
