I1 = imread('left.jpg');
I2 = imread('right.jpg');

[I1,I2,reprojectionMatrix] = rectifyStereoImages(I1,I2,stereoParams);

A = stereoAnaglyph(I1,I2);
figure
imshow(A)
title('Red-Cyan composite view of the rectified stereo pair image')

J1 = rgb2gray(I1);
J2 = rgb2gray(I2);

disparityRange = [32 128];
disparityMap = disparitySGM(J1,J2,'DisparityRange',disparityRange,'UniquenessThreshold',0);

figure
imshow(disparityMap,disparityRange)
title('Disparity Map')
colormap jet
colorbar