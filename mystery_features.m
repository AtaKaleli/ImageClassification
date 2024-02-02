function mysteryFeatures = mystery_features(images)
              
numberOfImages = numel(images);

%I created 907x1 cell for storing the features
mysteryFeatures = cell(numberOfImages, 1); 

% Extract LBP features from the images
    
    for i = 1:numberOfImages
        
        % Computing LBP features for each image patch
        lbp_features = extractLBPFeatures(images{i});
        % Storing the resulting feature vector in the mysteryFeatures cell
        mysteryFeatures{i} = lbp_features;
    end

%saving mysteryFeatures as  "mystery_Features.mat"
save("mystery_Features.mat", "mysteryFeatures", "-v7.3");


end


%%

%explanation of why I chose extractLBP features as a feature extractor.


% LBP is a potential feature extraction method for image classification that
% has several advantages, including efficiency, robustness to changes in 
% lighting and contrast, and the ability to capture important information 
% about the texture and local patterns of an image. By using LBP features, 
% I achieved good classification accuracy for a range of 
% image classification tasks, especially compared with features that are
% extracted using histograms. I also tried siftFeatures as an extraction
% method, but the results are not as good as features that are extracted
% using LBP. Only the LBP features gives me the + 70% accuracy for training
% accuracy. So this is the reason why I chose extractLBPFeatures.





