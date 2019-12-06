% %*************************************************************************************************************************
%                                               %Development dataset
%                                                    
% %*************************************************************************************************************************
%Defining image directories
Dataset = ["20Kph", "30Kph", "50Kph"];
for k = 1:length(Dataset)
    
    % loading images from a specified directory with the list of all '.jpg' files in the specified directory
    detectedSignsDir = Dataset{k};
    Data = fullfile(detectedSignsDir, '*.jpg');
    inputFileData = dir(Data);
    
    % loading images from GoldStandard directory with the list of all '.jpg' files in the specified directory
    goldDigitsDir = 'GoldStandards/';
    goldData = fullfile(goldDigitsDir, '*.jpg');
    goldInputFileData = dir(goldData);
    
    %initializing lists
    test_image = [];
    gold_image = [];
    edistance = [];
    
    %Creation of directory to store preprocessed images for training and testing     
    folder = '\\fs2\18231013\Embedded\Assignment3\test';
    %Check and create folder if folder doesn't exist
    if ~exist(folder, 'dir')
        mkdir(folder);
    end 
    
    %Creating subfolder with Speed as label for training and testing
    subfolder = fullfile('\\fs2\18231013\Embedded\Assignment3\test\',detectedSignsDir); 
    %Create subfolder if subfolder doesn't exist 
    if ~exist(subfolder, 'dir')
        mkdir(subfolder);
     end
     
    for i = 1:length(inputFileData)
        % Constructing file path and load image
        goldFilePath = fullfile(detectedSignsDir, inputFileData(i).name);
        I = imread(goldFilePath);
        
        %Standardizing image size for further processing
        imgROI = imresize(I, [450 450]);
        
        % adjusting the images for better detection of sign
        I = imadjust(imgROI,[.2 .3 0; .6 .7 1],[]);
        
        %converting image to grayscale , reducing dimension and complexity
        I = rgb2gray(I);
        
        % Binarizing the image
        bw = imbinarize(I);
        
        % Removing unwanted small objects from binarized image
        bw = bwareaopen(bw,70);
        
        %Constructing a disk-shaped structuring element with 6 as radius
        se = strel('disk',6);
        
        % Morphological closing by dilation on binarized image 
        bw = imclose(bw,se);
        
        %Tracing region boundaries where objects and holes are labeled.
        [B,L] = bwboundaries(bw);
        
        %suppressing the structures in image that are lighter than their surroundings and that are connected to the image border
        mask = imclearborder(bw);
        
        %Extracting first and second largest objects in the image
        mask = bwareafilt(mask, 2); 
        
        % standardizing image size for further processing
        mask = imresize(mask, [160 120]);
        % Returning 4 connected components
        cc = bwconncomp(mask, 4);
        %Creating rectangles containing region of intrests 
        stats = regionprops(cc, 'BoundingBox');
        
        %Checking for bounding box one and extracting region of interest
        if ~isempty(stats)
            
                %Considering first bounding box and masking region of interest 
                bbox = stats(1).BoundingBox;
                digitROI = mask(int16(bbox(2)):int16(bbox(2)+bbox(4)), int16(bbox(1)):int16(bbox(1)+bbox(3)), :);
                
                %cropping region of interest 
                digitROI = imcrop(digitROI, [7, 19, 55, 55]);
                
                if size(digitROI) ~= [0 0]
                    %Inverting the image to match with white pixelar digits in golden standard dataset
                    BW = imresize(imcomplement(digitROI), [160 120]);
                    
                    %Storing the complemented images 
                    test_image{i} = BW;
                    
                    %Resizing the images size for CNN model 
                    BW=imresize(BW, [160 160]);
                    
                    %Saving the images for testing and training of CNN model 
                    imwrite(BW, fullfile(subfolder, inputFileData(i).name));
                end             
        end
    end

    for i = 1:length(goldInputFileData)
        %Construct file path and load image
        goldFilePath = fullfile(goldDigitsDir, goldInputFileData(i).name);
        
        %loading images from GoldStandard directory with the list of all '.jpg' files in the specified directory
        I = imread(goldFilePath);
        
        %Standardizing image size with respect to test images
        I = imresize(I, [450 450]);
        
        %Extracting black digits by converting to YCbCr
        imageYCbCr = rgb2ycbcr(I);
        %Setting threshold limits for the 'y' channel and creating the mask
        mask = (imageYCbCr(:, :, 1) >= 0) & (imageYCbCr(:, :, 1) <= 20);
        
        %Cropping region of interests
        mask = imcrop(mask, [50   130  350  200]);
        
        %Standardizing image size w.r.t test images for processing
        mask = imresize(mask, [160 120]);
        
        %Storing processed golden standard images
        gold_image{i} = mask;

    end
    
    %Initializing true classification and false classification count to 0
    trueCount = 0;
    falseCount = 0;
     
    %Calculating Euclidian distance of test and gold standard images
    for i = 1:length(test_image)
        if ~isempty(test_image{i})
            for j = 1:length(goldInputFileData)
                %Calculating minimum Euclidian distance
                edistance(j) = sqrt(sum((test_image{i}(:) - gold_image{j}(:)) .^ 2));
                [~, index] = min(edistance);
            end
            
            %Comparing the classification output
            class = goldInputFileData(index).name(5:end-4);
            if class == detectedSignsDir(1:end-3)
             
            %Counting true positive and false positve counts
                trueCount = trueCount + 1;
            else
                falseCount = falseCount + 1;
            end
        end
    end
    
    fprintf("%s Dataset Classification: \n\t\t Correct images Identified %d \n\t\t Incorrect images identified are %d \n\t\t Accuracy is : %.2f\n", detectedSignsDir, (trueCount), (falseCount), (100 * trueCount) / (trueCount+falseCount));
    
end

%*************************************************************************************************************************
                                              %Training and testing in Convolutional Neural Network
                                                   
%*************************************************************************************************************************
%loading images from test directory
cnnDataSet = 'test';
imageData = imageDatastore(cnnDataSet, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%Counting label of each images
labelCount = countEachLabel(imageData);
I = readimage(imageData,25);
size(I);

%Loading and splitting the training and validation images randomly.
numTrainFiles = 175;
[imageTrain,imageValidate] = splitEachLabel(imageData,numTrainFiles,'randomize');

%Creation of Network layers
%First layer defines the size and type of the input data [160*160]
layers = [imageInputLayer([160 160 1])    
    
    %Creation of 3 Convolution2d layer with relu and maxpooling layer with
    %32 as number of channels in each filter and 2 as step size for traversal
    convolution2dLayer(5,32,'NumChannels',1);
    reluLayer();
    maxPooling2dLayer(2,'Stride',2);
    
    convolution2dLayer(5,32,'NumChannels', 32);
    reluLayer();
    maxPooling2dLayer(2,'Stride',2);
    
    convolution2dLayer(5,64)
    reluLayer();
    maxPooling2dLayer(2,'Stride',2);
    
    %Output layer with 3 outputs
    fullyConnectedLayer(3);
    softmaxLayer();
    classificationLayer()];

% Training the network with 0.01 as learning rate for 2 epoches 
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',2, ...
    'Shuffle','once', ...
    'Verbose',false);

%Creation of network
net = trainNetwork(imageTrain,layers,options);

%Testing the performance of the nework by evaluating accuracy on validation data
YPrediction = classify(net,imageValidate);
YValidation = imageValidate.Labels;

%Calculating accuracy
accuracy = sum(YPrediction == YValidation)/numel(YValidation);
fprintf("Accuracy of the model: %.2f", accuracy*100);

%*************************************************************************************************************************
                                                        %Stress dataset
                                                   
%*************************************************************************************************************************
%Loading images from a Stress dataset directory consisting of '.tif' files 
imagedir = 'stress_dataset';
Data = fullfile(imagedir, '*.TIF');
inputFileData = dir(Data);

%initializing the lists
stressTestImg = [];
test_image = [];
finalStressImg = [];

%Loading images from a GoldStandard dataset directory consisting of '.jpg' files 
goldDigitsDir = 'GoldStandards/';
goldData = fullfile(goldDigitsDir, '*.jpg');
goldInputFileData = dir(goldData);
gold_image = [];
edistance = [];

for i = 1:length(inputFileData)
    filePath = fullfile(imagedir, inputFileData(i).name);
    image = imread(filePath);
    %Resizing the image to standarize
    image_crop = imresize(image, [450 450]);
    
    %preprocessing of images as development dataset
    gray_img = rgb2gray(image);
    diff_im =  imsubtract(image(:,:,1), gray_img);
    diff_im = medfilt2(diff_im, [3,3]);
    diff_im = imbinarize(diff_im, 0.18);
    diff_im = bwareaopen(diff_im, 300);    
    diff_im = imresize(diff_im, [450 450]);
    %Finding the connected components and Boundingbox to extract circle
    cc = bwconncomp(diff_im);
    stats = regionprops(cc, 'BoundingBox');
    
    %Extracting circular region of interest with boundingbox 
    if length(stats) > 0  && length(stats) <= 5
   % figure(); subplot(1,3,1); imshow(image_crop);        
   % subplot(1,3,2); imshow(diff_im); 
        bbox = stats.BoundingBox;
        digitROI = image_crop(int16(bbox(2)):min(int16(bbox(2)+bbox(4)), 450),... 
        int16(bbox(1)):min(int16(bbox(1)+bbox(3)), 450), :);
        stressTestImg{i} = digitROI;
    % subplot(1,3,3); imshow(digitROI);
                
    elseif length(stats) > 5       
        % figure(); subplot(1,3,1); imshow(image_crop);
        % subplot(1,3,2); imshow(diff_im); 
        bbox = stats(4).BoundingBox;
        digitROI = image_crop(int16(bbox(2)):min(int16(bbox(2)+bbox(4)), 450),... 
        int16(bbox(1)):min(int16(bbox(1)+bbox(3)), 450), :);
        stressTestImg{i} = digitROI;
        %subplot(1,3,3); imshow(digitROI);
        
    else 
        img_crop = imcrop(image, [0,187, 450, 70]);
        gray_img = rgb2gray(img_crop);
        %figure(); subplot(1,3,1); imshow(image_crop);
        [m,n,o] = size(img_crop);
        diff_im =  imsubtract(img_crop(:,:,1), gray_img);
        diff_im = medfilt2(diff_im, [1,1]);
        diff_im = imbinarize(diff_im, 0.18);
        diff_im = bwareaopen(diff_im, 50); 
        %subplot(1,3,2); imshow(diff_im); 
        cc = bwconncomp(diff_im);
        stats = regionprops(cc, 'BoundingBox');

        if ~isempty(stats)
            bbox = stats.BoundingBox;
            digitROI = img_crop(int16(bbox(2)):min(int16(bbox(2)+bbox(4)), n), int16(bbox(1)):min(int16(bbox(1)+bbox(3)), n), :);
            % subplot(1,3,3); imshow(digitROI); 
            stressTestImg{i} = digitROI;
        end
    end
end

for i = 1:length(stressTestImg)
    
    I = stressTestImg{i};
    imgROI = imresize(I, [450 450]);
    I = imadjust(imgROI,[.2 .3 0; .6 .7 1],[]);
    I = rgb2gray(I);

    bw = imbinarize(I);
    bw = bwareaopen(bw,70);
    se = strel('disk',6);
    bw = imclose(bw,se);
    [B,L] = bwboundaries(bw);

    mask = imclearborder(bw);
    mask = bwareafilt(mask, 2); 
    mask = imresize(mask, [160 120]);
    cc = bwconncomp(mask, 4);
    stats = regionprops(cc, 'BoundingBox');
    if ~isempty(stats) 
        bbox = stats(1).BoundingBox;
        digitROI = mask(int16(bbox(2)):int16(bbox(2)+bbox(4)), int16(bbox(1)):int16(bbox(1)+bbox(3)), :);
        digitROI = imresize((digitROI), [160 120]);
        digitROI = imcrop(digitROI, [7, 37, 103, 84]);
        BW = imresize((digitROI), [160 120]);
        test_image{i} = imcomplement(BW);
        subplot(1,15, i); imshow(test_image{i});
        
        BW1 = imcomplement(imresize((digitROI), [160 160]));
        finalStressImg{i} = BW1;
    end   
end
figure();
for i = 1:length(goldInputFileData)
    filePath = fullfile(goldDigitsDir, goldInputFileData(i).name);
    I = imread(filePath);
    I = imresize(I, [450 450]);
    imageYCbCr = rgb2ycbcr(I);
    mask = (imageYCbCr(:, :, 1) >= 0) & (imageYCbCr(:, :, 1) <= 20);
    mask = imcrop(mask, [50   130  350  200]);
    mask = imresize(mask, [160 120]);
    gold_image{i} = mask;
end

for i = 1:length(test_image)
    if ~isempty(test_image{i})
        if size(test_image{i}) ~= [0 0]
            subplot(1,15,i); imshow(test_image{i});
            for j = 1:length(goldInputFileData)
                edistance(j) = sqrt(sum((test_image{i}(:) - gold_image{j}(:)) .^ 2));
                [~, index] = min(edistance);
            end
            %Prediction of sign
            answer = goldInputFileData(index).name(5:end-4);
            fprintf("Predicted Sign: \n %d - %s  \n",i, answer);
        end
    end
end

*********************************************
