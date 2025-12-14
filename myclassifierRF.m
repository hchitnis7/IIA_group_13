function yPred = myclassifierRF(I)
% MYCLASSIFIER - Predicts 4-digit captcha using Random Forest
% INPUT: I - RGB image
% OUTPUT: yPred - 1 x 4 vector of digits

persistent RFmodels trained img_size num_digits

if isempty(trained)
    % ---------------- Parameters ----------------
    numTrees = 100;
    img_size = [28 80];
    num_digits = 4;
    
    % ---------------- Load training data ----------------
    data = importdata('Train/labels.txt');
    img_nrs = data(:,1);
    true_labels = data(:,2:end);
    N = numel(img_nrs);
    
    X = [];
    Y = cell(num_digits,1);
    for d = 1:num_digits
        Y{d} = [];
    end
    
    % ---------------- Build feature matrix ----------------
    for n = 1:N
        im_train = imread(sprintf('Train/captcha_%04d.png', img_nrs(n)));
        feat = FeatureExtraction(im_train);
        X = [X; feat]; % size N x 2240
        
        labels = true_labels(n,:);
        % handle 3-digit CAPTCHAs
        if numel(labels) < num_digits
            labels = [zeros(1, num_digits - numel(labels)) labels];
        elseif numel(labels) > num_digits
            labels = labels(1:num_digits);
        end
        
        for d = 1:num_digits
            Y{d} = [Y{d}; labels(d)];
        end
    end
    
    % ---------------- Train Random Forest per digit ----------------
    RFmodels = cell(num_digits,1);
    for d = 1:num_digits
        RFmodels{d} = TreeBagger(numTrees,X,Y{d},'Method','classification','OOBPrediction','On');
    end
    
    trained = true;
end

% ---------------- Predict ----------------
yPred = zeros(1,num_digits);
feat = FeatureExtraction(I);

for d = 1:num_digits
    yPred_str = predict(RFmodels{d}, feat);
    yPred(d) = str2double(yPred_str);
end

% ---------------- Ensure length = 4 ----------------
if numel(yPred) < num_digits
    yPred = [zeros(1, num_digits - numel(yPred)) yPred];
elseif numel(yPred) > num_digits
    yPred = yPred(1:num_digits);
end

end
