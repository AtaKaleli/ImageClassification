function histFeatures = hist_features(images)
              

numberOfImages = numel(images);

%I created 908x1 cell for storing the features
histFeatures = cell(numberOfImages, 1); 


for i = 1:numberOfImages
    % imhist(I) displays a histogram for the intensity image I whose number
    % of bins are specified by the image type.  If I is a grayscale image,
    % imhist uses 256 bins as a default value.
    hist_counts = imhist(images{i});

    %computing the frequency of each intensity bin in the image.
    hist_freqs = hist_counts / sum(hist_counts); 
    
    % storing in feature matrix
    histFeatures{i, :} = hist_freqs';
end

%saving histFeatures as  "hist_features.mat"
save("hist_features.mat", "histFeatures", "-v7.3");

end