
function [BWfull, Gclean, info] = preprocessCaptcha(I, params)
% PREPROCESSCAPTCHA
%   CAPTCHA preprocessing only (no features, no classification)
%
% Outputs:
%   BWfull : final clean binary foreground mask (logical, deskewed)
%   Gclean : filtered grayscale image (double, [0,1], deskewed)
%   info   : diagnostic struct

    if nargin < 2 || isempty(params)
        params = default_preproc_params();
    end

    info = struct();

    %% 1) RGB → grayscale → double
    if size(I,3) == 3
        G = rgb2gray(I);
    else
        G = I;
    end
    G = im2double(G);
    

    %% 2) FFT + magnitude
    F  = fft2(G);
    Fs = fftshift(F);
    Mag = abs(Fs);

    [H,W] = size(G);
    cx = floor(W/2)+1;
    cy = floor(H/2)+1;

    %% 3) First symmetric FFT peak
    [pA, pA_sym] = find_first_symmetric_pair( ...
        Mag, cx, cy, params.r_min, params.r_max);

    peaks = [pA; pA_sym];

    %% 3b) Second symmetric FFT peak (optional)
    v = pA - [cx cy];
    [pB, pB_sym, found2] = find_second_pair_on_line( ...
        Mag, cx, cy, v, params.r_min, params.r_max, peaks);

    if found2
        peaks = [peaks; pB; pB_sym];
    end

    info.fftPeaks = peaks;
    info.nFFTpeaks = size(peaks,1);

    %% 4) Apply notch filter
    mask = true(H,W);
    mask = apply_notches(mask, peaks, params.notchRadius);

    Fs_filt = Fs .* mask;
    Gclean = real(ifft2(ifftshift(Fs_filt)));
    Gclean = mat2gray(Gclean);

    % contrast boost
    Gclean = imadjust(Gclean);

    %% 5) Spatial smoothing
    Gclean = medfilt2(Gclean, [params.med_size params.med_size]);
    % Gclean = imgaussfilt(Gclean, params.sigma);

    %% 6) Otsu threshold
    BW = imbinarize(Gclean, graythresh(Gclean));

    %% 7) Morphology 
    BW = ~BW;
    BW = imerode(BW, strel('disk', params.erodeDisk));
    BW = bwareaopen(BW, params.areaOpen1);
    BW = imdilate(BW, strel('disk', params.dilateDisk));

    %% 9) Distance-transform pruning (remove thin protrusions)
    D = bwdist(~BW);   % distance to background

    % % Threshold: remove pixels too far from main strokes
    % dtThresh = 1.7;
    % 
    % BW = BW & (D >= dtThresh);

    %% 8) Enforce 8-connected foreground
    CC = bwconncomp(BW, 8);
    stats = regionprops(CC, 'Area');

    if isempty(stats)
        BWfull = false(size(BW));
        info.failed = true;
        info.reason = "No foreground after morphology";
        return;
    end

    keepIdx = [stats.Area] >= params.minDigitArea;

    BWfull = false(size(BW));
    BWfull(cat(1, CC.PixelIdxList{keepIdx})) = true;

    


    %% Final close
    BWfull = imclose(BWfull, strel('disk', params.closeDisk));

    %% 10) Projection-based deskew
    angleDeg = estimate_skew_projection(BWfull, params);

    BWfull = imrotate(BWfull, angleDeg, 'bilinear', 'crop');
    Gclean = imrotate(Gclean, angleDeg, 'bilinear', 'crop');

    rows = any(BWfull,2);
    BWfull = BWfull(rows,:);
    Gclean = Gclean(rows,:);

    %% Info
    info.failed = false;
    info.skewAngleDeg = angleDeg;
end

    
    
function params = default_preproc_params()
    params = struct();

    % FFT
    params.r_min = 5;
    params.r_max = 100;
    params.notchRadius = 2;

    % Smoothing
    params.sigma = 0.15;
    params.med_size = 2;

    % Morphology
    params.erodeDisk  =3;
    params.dilateDisk = 2;
    params.closeDisk  = 1;

    params.areaOpen1     = 150;
    params.minDigitArea  = 200;

    % % Deskew (projection-based)
    % params.skew_angleRange = 50;    % degrees
    % params.skew_angleStep  = 1;  % degrees

    % Digit slicing / layout
    params.maxSlots        = 4;      % always output 4 slots
    params.minDigits       = 3;      % minimum digits
    params.forceLeading0   = true;   % 3-digit rule

end

function [p, psym] = find_first_symmetric_pair(Mag, cx, cy, r_min, r_max)
    [H,W] = size(Mag);
    [X,Y] = meshgrid(1:W,1:H);
    R = sqrt((X-cx).^2 + (Y-cy).^2);

    cand = (R>=r_min) & (R<=r_max);
    V = Mag; 
    V(~cand) = -Inf;

    [~, idx] = max(V(:));
    [y,x] = ind2sub([H,W], idx);
    p = [x y];

    xs = min(max(2*cx-x,1),W);
    ys = min(max(2*cy-y,1),H);
    psym = [xs ys];
end

function mask = apply_notches(mask, peaks, radius)
    [H,W] = size(mask);
    [X,Y] = meshgrid(1:W,1:H);
    for k = 1:size(peaks,1)
        x = peaks(k,1);
        y = peaks(k,2);
        disk = (X-x).^2 + (Y-y).^2 <= radius^2;
        mask(disk) = 0;
    end
end

function [p2, p2sym, found] = find_second_pair_on_line( ...
        Mag, cx, cy, v, r_min, r_max, excludePeaks)

    found = false;
    p2 = [NaN NaN]; p2sym = [NaN NaN];

    if norm(v) < 1
        return;
    end

    v = v / norm(v);
    [H,W] = size(Mag);

    t = (-max(H,W):max(H,W));
    xs = round(cx + t*v(1));
    ys = round(cy + t*v(2));

    valid = xs>=1 & xs<=W & ys>=1 & ys<=H;
    xs = xs(valid); ys = ys(valid);

    R = sqrt((xs-cx).^2 + (ys-cy).^2);
    keep = (R>=r_min & R<=r_max);

    xs = xs(keep); ys = ys(keep);

    scores = Mag(sub2ind([H,W], ys, xs));

    % suppress first peak
    for k=1:size(excludePeaks,1)
        dx = xs - excludePeaks(k,1);
        dy = ys - excludePeaks(k,2);
        scores(dx.^2+dy.^2 <= 4) = -Inf;
    end

    [m, idx] = max(scores);
    if isinf(m)
        return;
    end

    p2 = [xs(idx) ys(idx)];
    p2sym = [2*cx-p2(1), 2*cy-p2(2)];
    p2sym = min(max(p2sym,[1 1]),[W H]);
    found = true;
end

function angleDeg = estimate_skew_projection(BW, ~)
% FAST PCA-BASED DESKEW (no brute-force rotation)

    [y, x] = find(BW);

    if numel(x) < 50
        angleDeg = 0;
        return;
    end

    % Center coordinates
    x = x - mean(x);
    y = y - mean(y);

    % Covariance matrix
    C = cov(x, y);

    % Principal direction
    [V, D] = eig(C);

    [~, idx] = max(diag(D));
    v = V(:,idx);

    % Angle in degrees
    angleDeg = atan2d(v(2), v(1));

    % Convert to rotation angle
    angleDeg = angleDeg - 180 ;
end


% 
% function angleDeg = estimate_skew_projection(BW, params)
% % ESTIMATE_SKEW_PROJECTION
% % CAPTCHA-robust skew estimation using projection variance
% 
%     angleRange = params.skew_angleRange;
%     angleStep  = params.skew_angleStep;
% 
%     angles = -angleRange:angleStep:angleRange;
%     scores = zeros(size(angles));
% 
%     BW = logical(BW);
% 
%     for k = 1:numel(angles)
%         BWrot = imrotate(BW, angles(k), 'bilinear', 'crop');
% 
%         % Horizontal projection profile
%         proj = sum(BWrot, 2);
% 
%         % Variance measures alignment sharpness
%         scores(k) = var(proj);
%     end
% 
%     [~, idx] = max(scores);
%     angleDeg = angles(idx);
% end
