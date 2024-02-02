clear all; clc;
%sunrise89 invisible, copied sunrise90 , rename it as sunrise89, and pasted
%back into my dataset. I deleted problematic image names sunrise89

% label_images function convert the images to grayscale and form the class 
%labels

% It takes an input as dataset folder name, and gives output as images and
%their labels. I store the images and their labes as a cell, so I had 908x1
%cell for both images and labels after calling "label_images" function.

% Class label of images are simply the name of an image

folderName = "Dataset";
[images,labels] = label_images(folderName);


%this block is for calculating everything with using hist features

%extracts features of 908 images using histograms
histFeatures = hist_features(images);

%Form the training, validation, and testing sets.
[trainingSet,validationSet,testSet,trainingLabels,validationLabels,testLabels] = form_sets(histFeatures,labels);

%evaluates the accuracy of classification on the validation set and returns bestK value
bestK=training(trainingSet,validationSet,trainingLabels,validationLabels);

%evaluates and displays the accuracy of classification on the test set
accuracy=testing(bestK,trainingSet,testSet,trainingLabels,testLabels);

%---------------------------------------------------------------------------------------------------------------

%On the other hand, this block is for calculating everything with using 
% mystery features

 %extracts features of 908 images using LBP feature extractor method
 mysteryFeatures = mystery_features(images);
 
 %Form the training, validation, and testing sets.
 [trainingSet2,validationSet2,testSet2,trainingLabels2,validationLabels2,testLabels2] = form_sets(mysteryFeatures,labels);
 
 %evaluates the accuracy of classification on the validation set and returns bestK value
 bestK2=training(trainingSet2,validationSet2,trainingLabels2,validationLabels2);
 
 %evaluates and displays the accuracy of classification on the test set
 accuracy2=testing(bestK2,trainingSet2,testSet2,trainingLabels2,testLabels2);



%after the accuracies, it can be clearly seen that using hist features as
%a feature extraction is not a good choice.





















