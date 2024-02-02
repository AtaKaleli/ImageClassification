function [images,labels] = label_images(folderName)

%getting data and # of images in my dataset
files = dir(fullfile(folderName, "*.jpg"));
numberOfImages = numel(files);

%creating empty 908x1 cells for images and labels
images = cell(numberOfImages, 1);
labels = cell(numberOfImages, 1);

for i = 1:numberOfImages
    
    %getting the image and converting it to grayscale
    imagePath = fullfile(folderName, files(i).name);
    image = im2gray(imread(imagePath));
    
    %storing image in the images cell
    images{i} = image;

    % getting class label
    
    %normally, the format is [filepath, name, ext] = fileparts(files(i).name);
    %but as I dont use filepath and ext, matlab prefers using "~" instead of
    %filepath and ext
    [~, name, ~] = fileparts(files(i).name);
    name = regexprep(name, "[0-9]", "");
    labels{i} = name;
    
end

end