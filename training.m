function bestK = training(trainingSet,validationSet,trainingLabels,validationLabels)

%this part takes a bit long. So I try to explain my implementations as much
%as I can. First of all, for all K values 1 to 7, I have implemeted nested
%loops with my own implementation. I did not use build in function for
%to find the accuracy based on KNN values.


%finding #of training and validation set to use as dimensions in distanceArray
numberOfTraningSet = numel(trainingSet);
numberOfValidationSet = numel(validationSet);

%creating empty 227x454 distanceArray. This array holds every distance that
%will be calculated between validation and training sets. Good to remember
%that I have my features in training and validation sets.
distanceArray=zeros(numberOfValidationSet,numberOfTraningSet);


%here, I calculated the distance using pdist2 build in function. As I used
%cells, I need pairwise distance calculation.
for i = 1:numberOfValidationSet
    for j = 1:numberOfTraningSet
       distanceArray(i, j) = pdist2(validationSet{i},trainingSet{j});
    end
end



%%

%now this part is KNN=1 , which means for every row of my distance matrix,
%I need to take the minimum distance of 1 element and also it's index
%value. I did not use minimum distance in anywhere, but I need the minimum
%distance's index. After finding the index value, I used it as index value
%of trainingLabel, and compared it with validationLabel. For example, the
%minumum value distance's index is 356 out of 454. So for the first row, I
%have compared validationLabel{1} with trainingLabel{356}. If they are
%equal, I have incremented knn1_TrueMatches value, which is declared as 0
%initialy. After I compared all of the labels, I calculate the accuracy of
%the classification.

knn1_TrueMatches=0;

for i = 1:numberOfValidationSet
    
    [~, idx] = min(distanceArray(i, :));
    min1_DistanceIndex = idx;

    if strcmp(validationLabels{i},trainingLabels{min1_DistanceIndex})
        knn1_TrueMatches=knn1_TrueMatches+1;
    end

end

knn1_Accuracy=(knn1_TrueMatches/numberOfValidationSet)*100;




%%

%now in k=3, I need to take the minimum 3 of every row from my  distance
%matrix. First, I sort the distance array and get the first 3 indexes of 3
%minimum distances. After getting indexes, I created three labels, and I
%assigned each of them trainingLabels{minimumDistanceIndexes(1,2,3) and put
%these labels into "labels" cell. I continue commenting on the top after
%these part.

knn3_TrueMatches=0;

for i = 1:numberOfValidationSet
    
    % Find the 3 smallest  indices
    [~, index] = sort(distanceArray(i,:));
    min3_DistanceIndex = index(1:3);
    
    %find trainingLabels
    label1=trainingLabels{min3_DistanceIndex(1)};
    label2=trainingLabels{min3_DistanceIndex(2)};
    label3=trainingLabels{min3_DistanceIndex(3)};
    
     %create labels cell
    labels = {label1, label2, label3};
    
   % I converted labels to categorical array. This is because when I tried
   % not to use categorical , which is Arrays for categorical data, I got
   % an error. So after searching, I found this build in function.
    catLabels = categorical(labels);

    % I calculated the frequency count of each label. Here, my plan is that
    % If I want to do majority choicing( for example, I have 2 cloudy and 1 
    % shine label, the final  label must be cloudy), I need to
    % count the number of labels. So I can do this with using histograms.
    [counts, categories] = histcounts(catLabels, unique(catLabels));

    %After finding count of each categories(labels), I take the most
    % repeated element of this array as my majorityVoting result. 
    % This is perfectly works everytime, but when I have every label 
    % exist at the same time, for example cloudy,shine,sunrise, it takes
    %cloudy as a majorityVoting result. So instead of choosing randomly, I
    %continue getting the first element as majorityVoting result if there
    % exists an equality condition. To clarify better for you, 
    % let me give you two examples:
    
    %1)
    %I have shine,sunrise,sunrise as labels. After sorting based on
    %histograms, the counts become sunrise=2,shine=1. After
    %that, I take the maximum counted label as a majorityVoting.
    %2)
    %I have cloudy,shine,sunrise. As they all have count as 1, I take the
    %first one, predict that the image label is cloudy.

    [~, idx] = max(counts);
    mostFrequentLabel = categories(idx);
    
    %comparing validation label with mostFrequentLabel, which is the result
    %of majority voting.
    if strcmp(validationLabels{i},mostFrequentLabel)
        knn3_TrueMatches=knn3_TrueMatches+1;
    end
end

%I calculated the accuracy.
knn3_Accuracy=(knn3_TrueMatches/numberOfValidationSet)*100;


%%

%same idea with knn=3, I just have additional mininmum distance indexes and 
%labels here.
knn5_TrueMatches=0;

for i = 1:numberOfValidationSet
    
    % Find the 5 smallest indices
    [~, index] = sort(distanceArray(i,:));
    min5_DistanceIndex = index(1:5);
    
    %find trainingLabels
    label1=trainingLabels{min5_DistanceIndex(1)};
    label2=trainingLabels{min5_DistanceIndex(2)};
    label3=trainingLabels{min5_DistanceIndex(3)};
    label4=trainingLabels{min5_DistanceIndex(4)};
    label5=trainingLabels{min5_DistanceIndex(5)};
    
    %create labels cell
    labels = {label1, label2, label3, label4, label5};
    
    % convert to categorical array to use cell values
    catLabels = categorical(labels);

    % get the frequency count of each label
    [counts, categories] = histcounts(catLabels, unique(catLabels));

    % find the most frequent label(s)
    [~, idx] = max(counts);
    mostFrequentLabel = categories(idx);
    
    if strcmp(validationLabels{i},mostFrequentLabel)
        knn5_TrueMatches=knn5_TrueMatches+1;
    end
end

knn5_Accuracy=(knn5_TrueMatches/numberOfValidationSet)*100;



%%

%same idea with knn=3, I just have additional mininmum distance indexes and 
%labels here.
knn7_TrueMatches=0;

for i = 1:numberOfValidationSet
    
    % Find the 7 smallest  indices
    [~, index] = sort(distanceArray(i,:));
    min7_DistanceIndex = index(1:7);
    
    %find trainingLabels
    label1=trainingLabels{min7_DistanceIndex(1)};
    label2=trainingLabels{min7_DistanceIndex(2)};
    label3=trainingLabels{min7_DistanceIndex(3)};
    label4=trainingLabels{min7_DistanceIndex(4)};
    label5=trainingLabels{min7_DistanceIndex(5)};
    label6=trainingLabels{min7_DistanceIndex(6)};
    label7=trainingLabels{min7_DistanceIndex(7)};
    
    %create labels cell
    labels = {label1, label2, label3, label4, label5, label6, label7};
    
    
    % convert to categorical array to use cell values
    catLabels = categorical(labels);

    % got the frequency count of each label
    [counts, categories] = histcounts(catLabels, unique(catLabels));

    % find the most frequent label(s)
    [~, idx] = max(counts);
    mostFrequentLabel = categories(idx);
    
    if strcmp(validationLabels{i},mostFrequentLabel)
        knn7_TrueMatches=knn7_TrueMatches+1;
    end
end

knn7_Accuracy=(knn7_TrueMatches/numberOfValidationSet)*100;


%%

%after finding all accuracies, I have created 2 arrays to plot the results.
kValues=[1,3,5,7];
knnAccuracies=[knn1_Accuracy,knn3_Accuracy,knn5_Accuracy,knn7_Accuracy];

%At the and, I have four plotted figures. As I call training and test 
%function first with hist features, the first two plots are accuracy of
%classification with hist features and bestK value of hist Features. The
%remaining two plots  belongs to mystery.

figure;
plot(kValues, knnAccuracies, 'o-', 'LineWidth', 2);

xlabel('K Values');
ylabel('KNN Accuracies');
title('Accuracy of Classification on the Validation Set');

% Find the index of the maximum value in knnAccuracies
[~, maxIndex] = max(knnAccuracies);

% Use the index to return bestK value. I return the best K. I talked with
% Meryem hocam, she said that you dont need to enter the k value manually,
% so I assigned bestK based on the best knnAccuracy
bestK = kValues(maxIndex);




end

