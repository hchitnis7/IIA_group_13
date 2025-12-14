close all; clear;

data = importdata('Train/labels.txt');
img_nrs = data(:,1);
true_labels = data(:,2:5);  % 4 digits in file

train_patterns = [];
train_labels = {};

fprintf('Extracting training features...\n');

for i=1:numel(img_nrs)
    im = imread(sprintf('Train/captcha_%04d.png',img_nrs(i)));
    K = FeatureExtraction(im);

    if isempty(K), continue; end

    n = size(K,1);

    if n == 3
        lbl = true_labels(i,2:4); % skip leading zero
    else
        lbl = true_labels(i,:);
    end

    for j=1:n
        train_patterns(end+1,:) = K(j,:);
        train_labels{end+1} = num2str(lbl(j));
    end
end

train_labels = train_labels';

% -------- CLASSIFIER (same as your code) --------
tr = templateTree('MaxNumSplits',100);
Mdl = fitcensemble(double(train_patterns),train_labels,'Learners',tr);

save classifier Mdl
fprintf('Training done\n');
