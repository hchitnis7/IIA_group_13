%% demo_10_random_captchas.m
% Demo: Run classifier on 10 random captcha images and display predictions

clear; clc;

% ---------------- CONFIG ----------------
trainFolder = 'C:/Users/Harsh Chitnis/Desktop/IIA_group_13/Train';
numDemo = 10;  % number of images to demo

% ---------------- LIST IMAGES ----------------
imgFiles = dir(fullfile(trainFolder, 'captcha_*.png'));
if numel(imgFiles) < numDemo
    numDemo = numel(imgFiles);
end
idx = randperm(numel(imgFiles), numDemo);

% ---------------- LOAD CLASSIFIER ----------------
load('classifier.mat', 'Mdl');  % load trained model

% ---------------- CREATE FIGURE ----------------
figure('Name', sprintf('Demo: %d Random Captchas', numDemo), 'NumberTitle', 'off');

% ---------------- RUN CLASSIFIER ----------------
for i = 1:numDemo
    imPath = fullfile(trainFolder, imgFiles(idx(i)).name);
    I = imread(imPath);
    
    fprintf('Processing Image: %s\n', imgFiles(idx(i)).name);
    
    % Predict 4-digit label using your classifier
    digits = myclassifier(I);  % returns [d1 d2 d3 d4]
    
    % Display
    subplot(2,5,i);
    imshow(I);
    title(sprintf('%d %d %d %d', digits));
end
