s1im1 = im2double(im2gray(imread('S1-im1.png')));
s1im1 = imresize(s1im1, [750, 550]);
s2im1 = im2double(im2gray(imread('S2-im1.jpg')));
s2im1 = imresize(s2im1, [750, 550]);

total_time_fast = 0;
total_time_fastr = 0;

s1im1_color = im2double(imread('S1-im1.png'));
s1im1_color = imresize(s1im1_color, [750, 550]);
s2im1_color = im2double(imread('S2-im1.jpg'));
s2im1_color = imresize(s2im1_color, [750, 550]);


s1im2 = im2double(im2gray(imread('S1-im2.png')));
s1im2 = imresize(s1im2, [750, 550]);
s2im2 = im2double(im2gray(imread('S2-im2.jpg')));
s2im2 = imresize(s2im2, [750, 550]);

s1im2_color = im2double(imread('S1-im2.png'));
s1im2_color = imresize(s1im2_color, [750, 550]);
s2im2_color = im2double(imread('S2-im2.jpg'));
s2im2_color = imresize(s2im2_color, [750, 550]);


fast_s1im1 = my_fast_detector(s1im1, s1im1_color);
imwrite(fast_s1im1, 'fast_s1im1.png');
fastr_s1im1 = my_fastr(s1im1, fast_s1im1);
imwrite(fastr_s1im1, 'fastr_s1im1.png');
fast_s2im1 = my_fast_detector(s2im1, s2im1_color);
imwrite(fast_s2im1, 'fast_s2im1.png');
fastr_s2im1 = my_fastr(s2im1, fast_s2im1);
imwrite(fastr_s2im1, 'fastr_s2im1.png');
fast_s1im2 = my_fast_detector(s1im2, s1im2_color);
fastr_s1im2 = my_fastr(s1im2, fast_s1im2);
fast_s2im2 = my_fast_detector(s2im2, s2im2_color);
fastr_s2im2 = my_fastr(s2im2, fast_s2im2);

point_description(s1im1, s1im2, fast_s1im1, fast_s1im2);
point_description(s1im1, s1im2, fastr_s1im1, fastr_s1im2);
point_description(s2im1, s2im2, fast_s2im1, fast_s2im2);
point_description(s2im1, s2im2, fastr_s2im1, fastr_s2im2);



s2im3_color = imresize(im2double(imread('S2-im3.jpg')), [750, 550]);
s2im4_color = imresize(im2double(imread('S2-im4.jpg')), [750, 550]);

s3im1_color = imresize(im2double(imread('S3-im1.jpg')), [750, 550]);
s3im2_color = imresize(im2double(imread('S3-im2.jpg')), [750, 550]);
s3im3_color = imresize(im2double(imread('S3-im3.jpg')), [750, 550]);
s3im4_color = imresize(im2double(imread('S3-im4.jpg')), [750, 550]);

s4im1_color = imresize(im2double(imread('S4-im1.jpg')), [750, 550]);
s4im2_color = imresize(im2double(imread('S4-im2.jpg')), [750, 550]);
s4im3_color = imresize(im2double(imread('S4-im3.jpg')), [750, 550]);
s4im4_color = imresize(im2double(imread('S4-im4.jpg')), [750, 550]);

panorama_function_fastr2(s1im1_color, s1im2_color);
panorama_function_fastr(s2im1_color, s2im2_color, s2im3_color, s2im4_color);
panorama_function_fastr(s3im1_color, s3im2_color, s3im3_color, s3im4_color);
panorama_function_fastr(s4im1_color, s4im2_color, s4im3_color, s4im4_color);

function keypoints = my_fast_detector(image, color)
tic;
    % Parameters
    n = 5; % Number of contiguous pixels to consider
    t = 0.3; % Threshold value

    padded_image = padarray(image, [3, 3], 'both', 'replicate');

    % High-speed test
    brighter = padded_image(4:end-3, 4:end-3) + t - padded_image(1:end-6, 4:end-3);
    darker = padded_image(4:end-3, 4:end-3) - t - padded_image(1:end-6, 4:end-3);

    contiguous_brighter = (brighter & circshift(brighter, -1, 2)) | ...
                          (circshift(brighter, -1, 2) & circshift(brighter, -2, 2)) | ...
                          (circshift(brighter, -2, 2) & circshift(brighter, -3, 2));

    contiguous_darker = (darker & circshift(darker, -1, 2)) | ...
                        (circshift(darker, -1, 2) & circshift(darker, -2, 2)) | ...
                        (circshift(darker, -2, 2) & circshift(darker, -3, 2));

    high_speed_test_passed = (sum(contiguous_brighter(:)) >= 3) | (sum(contiguous_darker(:)) >= 3);

    % Check if high-speed test passed
    if high_speed_test_passed
        % Apply the full segment test
        keypoints = full_segment_test(image, t, n);
    else
        keypoints = false(size(image));
    end
    % figure
    % imshow(color);
    % hold on;
    % if any(keypoints(:))
    %     [row, col] = find(keypoints);
    %     scatter(col, row, 1, 'g', 'filled');
    % end
    % hold off;
    fprintf('Fast time: %4.3f seconds\n', toc)
end



function keypoints = full_segment_test(image, t, n)
    [rows, cols] = size(image);
    keypoints = false(rows, cols);

    for i = 4:rows-3
        for j = 4:cols-3
            center_intensity = image(i, j);
            
            if is_corner(image(i-3:i+3, j-3:j+3), center_intensity, t, n)
                keypoints(i, j) = true;
            end
        end
    end
end



function is_corner_pixel = is_corner(pixels, center_intensity, t, n)
    brighter_count = sum(pixels(:) > center_intensity + t);
    darker_count = sum(pixels(:) < center_intensity - t);
    
    is_corner_pixel = brighter_count >= n || darker_count >= n;
end



function my_fastr = my_fastr(image, fast_image)
tic;
    k = 0.01;
    sobel = [-1 0 1; -2 0 2; -1 0 1];
    gaus = fspecial('gaussian',20,3);
    dog = conv2(gaus, sobel);
    ix = imfilter(image, dog);
    iy = imfilter(image, dog');

    ix2 = ix .* ix;
    iy2 = iy .* iy;
    ixy = ix .* iy;

    ix2_smooth = imfilter(ix2, gaus);
    iy2_smooth = imfilter(iy2, gaus);
    ixy_smooth = imfilter(ixy, gaus);

    det_M = ix2_smooth .* iy2_smooth - ixy_smooth.^2;
    trace_M = ix2_smooth + iy2_smooth;
    harcor = det_M - k * trace_M.^2;

    thresh = 0.00001;
    my_fastr = fast_image & harcor > thresh;
    
    % figure
    % imshow(image);
    % hold on;
    % [row, col] = find(my_fastr);
    % scatter(col, row, 1, 'r', 'filled');
    % hold off;
    fprintf('FastR time: %4.3f seconds\n', toc)
end


function point_description(s1, s2, s1_fast, s2_fast)
    [I_fast_x, I_fast_y] = find(s1_fast);
    points_s1 = [I_fast_y, I_fast_x];
    [features_s1, points_s1] = extractFeatures(s1, points_s1, 'Method', 'SURF');

    [I_fast2_x, I_fast2_y] = find(s2_fast);
    points_s2 = [I_fast2_y, I_fast2_x];
    [features_s2, points_s2] = extractFeatures(s2, points_s2, 'Method', 'SURF');

    indexPairs = matchFeatures(features_s1, features_s2, 'Unique', true);
    matchedPoints1 = points_s1(indexPairs(:,1), :);
    matchedPoints2 = points_s2(indexPairs(:,2), :);
    figure
    match = showMatchedFeatures(s1, s2, matchedPoints1, matchedPoints2, 'montage');
    title('Matched Features');
end



function panorama_function_fastr(s1, s2, s3, s4)

buildingScene = {s1, s2, s3, s4};

montage(buildingScene)

I = buildingScene{1};

grayImage = im2gray(I);
I_fast = my_fast_detector(grayImage, I);
I_fastr = my_fastr(I, I_fast);
[I_fast_x, I_fast_y] = find(I_fastr);
points = [I_fast_y, I_fast_x];
[features, points] = extractFeatures(grayImage, points);

numImages = numel(buildingScene);
tforms(numImages) = projtform2d;

imageSize = zeros(numImages,2);

for n = 2:numImages
    pointsPrevious = points;
    featuresPrevious = features;

    I = buildingScene{n};

    grayImage = im2gray(I);

    imageSize(n,:) = size(grayImage);

    I_fast = my_fast_detector(grayImage, I);
    I_fastr = my_fastr(I, I_fast);
    [I_fast_x, I_fast_y] = find(I_fastr);
    points = [I_fast_y, I_fast_x];
    [features, points] = extractFeatures(grayImage, points);

    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);

    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);

    tforms(n) = estgeotform2d(matchedPoints, matchedPointsPrev,...
        'projective', 'Confidence', 99.0, 'MaxNumTrials', 30);

    tforms(n).T = tforms(n-1).T * tforms(n).T;
end

for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

avgXLim = mean(xlim, 2);
[~,idx] = sort(avgXLim);
centerIdx = floor((numel(tforms)+1)/2);
centerImageIdx = idx(centerIdx);

Tinv = invert(tforms(centerImageIdx));
for i = 1:numel(tforms)
    tforms(i).T = Tinv.T * tforms(i).T;
end

for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

maxImageSize = max(imageSize);

xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

width  = round(xMax - xMin);
height = round(yMax - yMin);

panorama = zeros([height width 3], 'like', I);

blender = vision.AlphaBlender('Operation', 'Binary mask', 'MaskSource', 'Input port');

xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

for i = 1:numImages
    I = buildingScene{i};

    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);

    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);

    panorama = step(blender, panorama, warpedImage, mask);
end

figure
imshow(panorama)
figure

end



function panorama_function_fastr2(s1, s2)

buildingScene = {s1, s2};

montage(buildingScene)

I = buildingScene{1};

grayImage = im2gray(I);
I_fast = my_fast_detector(grayImage, I);
I_fastr = my_fastr(I, I_fast);
[I_fast_x, I_fast_y] = find(I_fastr);
points = [I_fast_y, I_fast_x];
[features, points] = extractFeatures(grayImage, points);

numImages = numel(buildingScene);
tforms(numImages) = projtform2d;

imageSize = zeros(numImages,2);

for n = 2:numImages
    pointsPrevious = points;
    featuresPrevious = features;

    I = buildingScene{n};

    grayImage = im2gray(I);

    imageSize(n,:) = size(grayImage);

    I_fast = my_fast_detector(grayImage, I);
    I_fastr = my_fastr(I, I_fast);
    [I_fast_x, I_fast_y] = find(I_fastr);
    points = [I_fast_y, I_fast_x];
    [features, points] = extractFeatures(grayImage, points);

    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);

    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);

    tforms(n) = estgeotform2d(matchedPoints, matchedPointsPrev,...
        'projective', 'Confidence', 99.0, 'MaxNumTrials', 30);

    tforms(n).T = tforms(n-1).T * tforms(n).T;
end

for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

avgXLim = mean(xlim, 2);
[~,idx] = sort(avgXLim);
centerIdx = floor((numel(tforms)+1)/2);
centerImageIdx = idx(centerIdx);

Tinv = invert(tforms(centerImageIdx));
for i = 1:numel(tforms)
    tforms(i).T = Tinv.T * tforms(i).T;
end

for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

maxImageSize = max(imageSize);

xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

width  = round(xMax - xMin);
height = round(yMax - yMin);

panorama = zeros([height width 3], 'like', I);

blender = vision.AlphaBlender('Operation', 'Binary mask', 'MaskSource', 'Input port');

xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

for i = 1:numImages
    I = buildingScene{i};

    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);

    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);

    panorama = step(blender, panorama, warpedImage, mask);
end

figure
imshow(panorama)
figure

end



function panorama_function_fast(s1, s2, s3, s4)

buildingScene = {s1, s2, s3, s4};

montage(buildingScene)

I = buildingScene{1};

grayImage = im2gray(I);
I_fast = my_fast_detector(grayImage, I);
[I_fast_x, I_fast_y] = find(I_fast);
points = [I_fast_y, I_fast_x];
[features, points] = extractFeatures(grayImage, points);

numImages = numel(buildingScene);
tforms(numImages) = projtform2d;

imageSize = zeros(numImages,2);

for n = 2:numImages
    pointsPrevious = points;
    featuresPrevious = features;

    I = buildingScene{n};

    grayImage = im2gray(I);

    imageSize(n,:) = size(grayImage);

    I_fast = my_fast_detector(grayImage, I);
    [I_fast_x, I_fast_y] = find(I_fast);
    points = [I_fast_y, I_fast_x];
    [features, points] = extractFeatures(grayImage, points);

    indexPairs = matchFeatures(features, featuresPrevious, 'Unique', true);

    matchedPoints = points(indexPairs(:,1), :);
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);

    tforms(n) = estgeotform2d(matchedPoints, matchedPointsPrev,...
        'projective', 'Confidence', 80.0, 'MaxNumTrials', 30);

    tforms(n).T = tforms(n-1).T * tforms(n).T;
end

for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

avgXLim = mean(xlim, 2);
[~,idx] = sort(avgXLim);
centerIdx = floor((numel(tforms)+1)/2);
centerImageIdx = idx(centerIdx);

Tinv = invert(tforms(centerImageIdx));
for i = 1:numel(tforms)
    tforms(i).T = Tinv.T * tforms(i).T;
end

for i = 1:numel(tforms)
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]);
end

maxImageSize = max(imageSize);

xMin = min([1; xlim(:)]);
xMax = max([maxImageSize(2); xlim(:)]);

yMin = min([1; ylim(:)]);
yMax = max([maxImageSize(1); ylim(:)]);

width  = round(xMax - xMin);
height = round(yMax - yMin);

panorama = zeros([height width 3], 'like', I);

blender = vision.AlphaBlender('Operation', 'Binary mask', 'MaskSource', 'Input port');


xLimits = [xMin xMax];
yLimits = [yMin yMax];
panoramaView = imref2d([height width], xLimits, yLimits);

for i = 1:numImages
    I = buildingScene{i};
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView', panoramaView);
    panorama = step(blender, panorama, warpedImage, mask);
end

figure
imshow(panorama)
figure

end




