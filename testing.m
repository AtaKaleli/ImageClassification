function accuracy = testing(bestK,trainingSet,testSet,trainingLabels,testLabels)

%I as did not used build in function, based on the bestK value entered from
%the main, I entered if condition that satisfies the equation.

%As testing process are almost same with training(little changes like
%comparing training set with test set etc.) I copied what I did in the
%training function. As I did this, I delete all of the comments, as I
%explained all my work in a very detailed way in the training function.

numberOfTraningSet = numel(trainingSet);
numberOfTestSet = numel(testSet);

distanceArray=zeros(numberOfTestSet,numberOfTraningSet);


for i = 1:numberOfTestSet
    for j = 1:numberOfTraningSet
       distanceArray(i, j) = pdist2(testSet{i},trainingSet{j});
    end
end



if bestK == 1
    
    knn1_TrueMatches=0;

    for i = 1:numberOfTestSet
        
        [~, idx] = min(distanceArray(i, :));
        min1_DistanceIndex = idx;
    
        if strcmp(testLabels{i},trainingLabels{min1_DistanceIndex})
            knn1_TrueMatches=knn1_TrueMatches+1;
        end
    
    end
    
    accuracy=(knn1_TrueMatches/numberOfTestSet)*100;



elseif bestK==3
   
    knn3_TrueMatches=0;
    
    for i = 1:numberOfTestSet
        
        [~, index] = sort(distanceArray(i,:));
        min3_DistanceIndex = index(1:3);
        
        label1=trainingLabels{min3_DistanceIndex(1)};
        label2=trainingLabels{min3_DistanceIndex(2)};
        label3=trainingLabels{min3_DistanceIndex(3)};
        
        labels = {label1, label2, label3};
       
        catLabels = categorical(labels);
        [counts, categories] = histcounts(catLabels, unique(catLabels));
        [~, idx] = max(counts);
        mostFrequentLabel = categories(idx);
        
        if strcmp(testLabels{i},mostFrequentLabel)
            knn3_TrueMatches=knn3_TrueMatches+1;
        end

    end
    
    accuracy=(knn3_TrueMatches/numberOfTestSet)*100;


elseif bestK==5
    
    knn5_TrueMatches=0;
    
    for i = 1:numberOfTestSet
        
        [~, index] = sort(distanceArray(i,:));
        min5_DistanceIndex = index(1:5);
        
        label1=trainingLabels{min5_DistanceIndex(1)};
        label2=trainingLabels{min5_DistanceIndex(2)};
        label3=trainingLabels{min5_DistanceIndex(3)};
        label4=trainingLabels{min5_DistanceIndex(4)};
        label5=trainingLabels{min5_DistanceIndex(5)};
        
        labels = {label1, label2, label3, label4, label5};
        catLabels = categorical(labels);
        [counts, categories] = histcounts(catLabels, unique(catLabels));
        [~, idx] = max(counts);
        mostFrequentLabel = categories(idx);
        
        if strcmp(testLabels{i},mostFrequentLabel)
            knn5_TrueMatches=knn5_TrueMatches+1;
        end
    
    end
    
    accuracy=(knn5_TrueMatches/numberOfTestSet)*100;


elseif bestK==7
    
    knn7_TrueMatches=0;
    
    for i = 1:numberOfTestSet
        
        [~, index] = sort(distanceArray(i,:));
        min7_DistanceIndex = index(1:7);
        
        label1=trainingLabels{min7_DistanceIndex(1)};
        label2=trainingLabels{min7_DistanceIndex(2)};
        label3=trainingLabels{min7_DistanceIndex(3)};
        label4=trainingLabels{min7_DistanceIndex(4)};
        label5=trainingLabels{min7_DistanceIndex(5)};
        label6=trainingLabels{min7_DistanceIndex(6)};
        label7=trainingLabels{min7_DistanceIndex(7)};
        
        labels = {label1, label2, label3, label4, label5, label6, label7};
        
        catLabels = categorical(labels);
        [counts, categories] = histcounts(catLabels, unique(catLabels));
        [~, idx] = max(counts);
        mostFrequentLabel = categories(idx);
        
        if strcmp(testLabels{i},mostFrequentLabel)
            knn7_TrueMatches=knn7_TrueMatches+1;
        end
    
    end
    
    accuracy=(knn7_TrueMatches/numberOfTestSet)*100;
end


figure;
plot(bestK, accuracy, 'o-', 'LineWidth', 2);
xlabel('Best K Value');
ylabel('Accuracy');
title(['Accuracy of Classification on the Test Set with K:', num2str(bestK)]);



end




