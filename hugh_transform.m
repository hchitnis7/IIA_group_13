function lineMask = hugh_transform(im)
    [H,theta,rho] = hough(im);
    P = houghpeaks(H, 25, 'Threshold', ceil(0.35*max(H(:))));
    L = houghlines(im, theta, rho, P, 'FillGap', 9, 'MinLength', 50);

    [M,N] = size(im);
    lineMask = false(M,N);
    halfWidth = 2;

    for k = 1:numel(L)
        p1 = L(k).point1;
        p2 = L(k).point2;

        v = double(p2 - p1);
        nv = norm(v);
        if nv < 1, continue; end

        n = halfWidth * [-v(2), v(1)] / nv;
        rect = [p1+n; p1-n; p2-n; p2+n];
        lineMask = lineMask | poly2mask(rect(:,1), rect(:,2), M, N);
    end
end

% function lines = hugh_transform(im, FG)
% 
%     [H,theta,rho] = hough(im);
%     P = houghpeaks(H, 25, 'Threshold', ceil(0.35*max(H(:))));
%     L = houghlines(im, theta, rho, P, 'FillGap', 9, 'MinLength', 50);
% 
%     [M,N] = size(FG);
%     lineMask = false(M,N);
%     halfWidth = 2;
% 
%     for k = 1:numel(L)
%         p1 = L(k).point1;
%         p2 = L(k).point2;
% 
%         v = double(p2 - p1);
%         nv = norm(v);
%         if nv < 1, continue; end
% 
%         n = halfWidth * [-v(2), v(1)] / nv;
%         rect = [p1+n; p1-n; p2-n; p2+n];
%         lineMask= lineMask | poly2mask(rect(:,1), rect(:,2), M, N);
%     end
% 
%     lines = lineMask;
% end
