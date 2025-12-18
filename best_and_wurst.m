clc; clear;
tic

% -------------------------------------------------
% Load labels
% -------------------------------------------------
data = importdata('Train/labels.txt');
img_nrs = data(:,1);
true_labels = data(:,2:end);

N = size(img_nrs,1);
my_labels = zeros(size(true_labels));

BWmasks = cell(N,1);

% -------------------------------------------------
% Run classifier + preprocessing
% -------------------------------------------------
for n = 1:N
    k = img_nrs(n);
    Image = imread(sprintf('Train/captcha_%04d.png', k));

    % Get binary mask
    [BW, ~, ~] = preprocessCaptcha(Image);
    BWmasks{n} = BW;

    % Classify
    my_labels(n,:) = myclassifier(Image);
end

% -------------------------------------------------
% Compute errors
% -------------------------------------------------
diffMat  = abs(true_labels - my_labels);
errCount = sum(diffMat ~= 0, 2);
correct  = errCount == 0;

fprintf('\nAccuracy: %.4f\n', mean(correct));

% -------------------------------------------------
% Select best & worst
% -------------------------------------------------
bestIdx  = find(correct);
worstIdx = find(~correct);

bestIdx = bestIdx(1:min(10,numel(bestIdx)));

[~, w]   = sort(errCount(worstIdx), 'descend');
worstIdx = worstIdx(w(1:min(10,numel(w))));

% -------------------------------------------------
% Visualize BEST samples
% -------------------------------------------------
figure('Name','Top 10 BEST classified','Color','w');

for i = 1:numel(bestIdx)
    n = bestIdx(i);
    k = img_nrs(n);

    Image = imread(sprintf('Train/captcha_%04d.png', k));

    subplot(2,5,i);
    imshowpair(Image, BWmasks{n}, 'montage');
    axis off;

    title(sprintf( ...
        'ID %04d\nTrue: %s\nPred: %s\n', ...
        k, ...
        num2str(true_labels(n,:)), ...
        num2str(my_labels(n,:))), ...
        'FontSize', 9);
end

% -------------------------------------------------
% Visualize WORST samples
% -------------------------------------------------
figure('Name','Top 10 WORST classified','Color','w');

for i = 1:numel(worstIdx)
    n = worstIdx(i);
    k = img_nrs(n);

    Image = imread(sprintf('Train/captcha_%04d.png', k));

    subplot(2,5,i);
    imshowpair(Image, BWmasks{n}, 'montage');
    axis off;

    title(sprintf( ...
        'ID %04d\nTrue: %s\nPred: %s\n', ...
        k, ...
        num2str(true_labels(n,:)), ...
        num2str(my_labels(n,:))), ...
        'FontSize', 9);
end

toc
