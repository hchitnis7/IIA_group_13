function bw = preprocess_captcha(impath)

% ----- 1) Read image -----
if ischar(impath) || isstring(impath)
    I = imread(impath);
else
    I = impath;
end
Igray = rgb2gray(I);
Igray = im2double(Igray);

% ----- 2) Remove diagonal banding using Wiener -----
I1 = wiener2(Igray, [7 7]);   % suppress large-scale stripes

% ----- 3) Median filter to kill color speckle -----
I2 = medfilt2(I1, [3 3]);

% ----- 4) Bilateral smoothing (keeps edges, kills noise) -----
I3 = imbilatfilt(I2, 0.2, 5);

% ----- 5) Strong adaptive threshold (Sauvola) -----
T = adaptthresh(I3, 0.55, 'ForegroundPolarity','dark', ...
                'Statistic','mean');    
bw = imbinarize(I3, T);

% ----- 6) Remove tiny specks but KEEP thin strokes -----
bw = bwareaopen(bw, 15);   % remove VERY small noise only

% ----- 7) Final cleanup (no closing/opening – preserves digits) -----
bw = imfill(bw, 'holes');

end
