function [skeleton_im, BW2] = morph(im)
%Apply adaptive binarization and performs open and skeleton. Number 300
% Can be changed. Higher number more aggresive removal.
BW  = imbinarize(im,'adaptive','ForegroundPolarity','dark','Sensitivity',0.55);
BW2 = medfilt2(BW, [5 5]);

FG  = ~BW2;
FG  = bwareaopen(FG, 300);
BW2 = ~FG;

FG = ~BW2;

skeleton_im = bwmorph(FG, 'thin', Inf);
end