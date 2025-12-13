function destripped = remove_strips(im)

I = im2double(im);
if ndims(I)==3, I = rgb2gray(I); end
[M,N] = size(I);

F0 = fftshift(fft2(I));

[u,v] = meshgrid((-N/2):(N/2-1), (-M/2):(M/2-1));

theta = -20*pi/180;
d = abs(u*cos(theta) + v*sin(theta));

D0 = 6;
H = 1 - exp(-(d.^2)/(2*D0^2));

I2 = real(ifft2(ifftshift(F0 .* H)));
I2 = mat2gray(I2);

destripped = I2;

% destripped = imnlmfilt(J, ...
%     'SearchWindowSize',21, ...
%     'ComparisonWindowSize',7, ...
%     'DegreeOfSmoothing',0.12);

end