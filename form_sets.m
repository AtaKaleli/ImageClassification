function [trainingSet,validationSet,testSet,trainingLabels,validationLabels,testLabels] = form_sets(features,labels)

%first of all, I have calculated all the indexes that I need for in "for loop"
%by handwriting.

%creating trainingSet
%the size of the training set is 150(cloud)+126(shine)+178(sunrise)=454
%these numbers comes from the first 50% of each class
trainingSet=cell(454,1);
trainingLabels=cell(454,1);

%this is the counter of trainingSet index. It goes from 1 to 454
trainCount=1;

%taking first 50% of cloud image group
for i = 1:150
      trainingSet{trainCount}=features{i}; 
      trainingLabels{trainCount}=labels{i};
      trainCount=trainCount+1;
end

%taking first 50% of shine image group
for i = 301:426
      trainingSet{trainCount}=features{i};
      trainingLabels{trainCount}=labels{i};
      trainCount=trainCount+1;
end

%taking first 50% of sunrise image group
for i = 553:730
      trainingSet{trainCount}=features{i};
      trainingLabels{trainCount}=labels{i};
      trainCount=trainCount+1;
end


%creating validationSet
%the size of the validation set is 75(cloud)+63(shine)+89(sunrise)=227
%these numbers comes from the remaining 25% of each class
validationSet=cell(227,1);
validationLabels=cell(227,1);

%this is the counter of validationSet index. It goes from 1 to 227
validationCount=1;


%the remaining 25% of cloud class
for i = 151:225
      validationSet{validationCount}=features{i};
      validationLabels{validationCount}=labels{i};
      validationCount=validationCount+1;
end

%the remaining 25% of shine class
for i = 427:489
      validationSet{validationCount}=features{i};
      validationLabels{validationCount}=labels{i};
      validationCount=validationCount+1;
end

%the remaining 25% of sunrise class
for i = 731:819
      validationSet{validationCount}=features{i};
      validationLabels{validationCount}=labels{i};
      validationCount=validationCount+1;
end


%creating testSet
%the size of the test set is 75(cloud)+63(shine)+89(sunrise)=227
%these numbers comes from the last remaining 25% of each class
testSet=cell(227,1);
testLabels=cell(227,1);

%this is the counter of testSet index. It goes from 1 to 227
testCount=1;

%the last remaining 25% of cloud class
for i = 226:300
      testSet{testCount}=features{i};
      testLabels{testCount}=labels{i};
      testCount=testCount+1;
end

%the last remaining 25% of shine class
for i = 490:552
      testSet{testCount}=features{i};
      testLabels{testCount}=labels{i};
      testCount=testCount+1;
end

%the last remaining 25% of sunrise class
for i = 820:908
      testSet{testCount}=features{i};
      testLabels{testCount}=labels{i};
      testCount=testCount+1;
end

end