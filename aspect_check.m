clc; clear;

% -------------------------------------------------
% Load labels
% -------------------------------------------------
data = importdata('Train/labels.txt');
img_nrs = data(:,1);
labels  = data(:,2:end);

N = size(labels,1);

% Determine 3-digit vs 4-digit from labels
isThree = labels(:,1) == 0;   % leading zero → 3-digit
isFour  = ~isThree;

% Storage
aspectRatio = nan(N,1);
bboxW = nan(N,1);
bboxH = nan(N,1);
failed = false(N,1);

% -------------------------------------------------
% Loop over images
% -------------------------------------------------
for n = 1:N
    k = img_nrs(n);
    I = imread(sprintf('Train/captcha_%04d.png', k));

    [BW, ~, info] = preprocessCaptcha(I);

    if isfield(info,'failed') && info.failed
        failed(n) = true;
        continue;
    end

    [y, x] = find(BW);
    if isempty(x)
        failed(n) = true;
        continue;
    end

    w = max(x) - min(x) + 1;
    h = max(y) - min(y) + 1;

    bboxW(n) = w;
    bboxH(n) = h;
    aspectRatio(n) = w / h;
end

% -------------------------------------------------
% Remove failed samples
% -------------------------------------------------
valid = ~failed;

AR3 = aspectRatio(valid & isThree);
AR4 = aspectRatio(valid & isFour);

fprintf('Valid samples: %d / %d\n', sum(valid), N);
fprintf('3-digit samples: %d\n', numel(AR3));
fprintf('4-digit samples: %d\n', numel(AR4));

% -------------------------------------------------
% Statistics
% -------------------------------------------------
fprintf('\nAspect Ratio Statistics:\n');
fprintf('3-digit: mean = %.2f | std = %.2f\n', mean(AR3), std(AR3));
fprintf('4-digit: mean = %.2f | std = %.2f\n', mean(AR4), std(AR4));

% -------------------------------------------------
% Visualization
% -------------------------------------------------
figure('Color','w');

subplot(1,2,1);
histogram(AR3, 30, 'Normalization','probability');
title('3-digit captchas');
xlabel('Bounding box aspect ratio (W/H)');
ylabel('Probability');
grid on;

subplot(1,2,2);
histogram(AR4, 30, 'Normalization','probability');
title('4-digit captchas');
xlabel('Bounding box aspect ratio (W/H)');
ylabel('Probability');
grid on;

sgtitle('Aspect Ratio Distribution: 3-digit vs 4-digit');
