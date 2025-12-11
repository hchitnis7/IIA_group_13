function S = myclassifier(im)

% This classifier is not state of the art... but should give you an idea of
% the format we expect to make it easy to keep track of your scores. 
%
% Input is the image, output is a 1 x 4 vector of the three or four numbers 
% visible in the image, where, if there are only three numbers, pad the 
% output with a zero to the left; i.e., 123 => [0,1,2,3].
%
% This baseline classifier tries to guess... so should score on average
% about: 1/2*1/2 * 1/3^3 + 1/2*1/2 * 1/3^4 = 0.01234567, 
% A 1.2% chance of guessing the correct answer. 


if (rand <= 0.5) % Three digits
    S=[0, randi([3,5],[1,3])]; % Padding with a zero to the left
else % Four digits
    S=randi([3,5],[1,4]);
end
