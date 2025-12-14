tic

data = importdata('Train/labels.txt');
img_nrs = data(:,1);
true_labels = data(:,2:end);
N = numel(img_nrs);

my_labels = zeros(N,4); % always 4 digits

for n = 1:N
    k = img_nrs(n);
    im = imread(sprintf('Train/captcha_%04d.png',k));
    my_labels(k,:) = myclassifierRF(im);
end

% Prepend zeros for 3-digit CAPTCHAs
for n = 1:N
    labels = true_labels(n,:);
    if numel(labels) < 4
        true_labels(n,:) = [zeros(1,4-numel(labels)) labels];
    elseif numel(labels) > 4
        true_labels(n,:) = labels(1:4);
    end
end

% Compute accuracy
correct = sum(abs(true_labels - my_labels),2) == 0;
fprintf('Overall Accuracy: %f\n', mean(correct));

toc
