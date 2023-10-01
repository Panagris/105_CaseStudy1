close all;
load("COVIDbyCounty.mat");
%% 3.2 Clustering
% rng("default");
k = 18;
r = randperm(size(CNTY_COVID,1)); % permute row numbers
covidShuffled = CNTY_COVID(r,:);
cntyShuffled = CNTY_CENSUS.DIVISION(r,:);

trainingGroup = covidShuffled(1:180,:);
testingGroup = covidShuffled(181:225,:);
centroidLabels = 1:k;

[idx2, centroids] = kmeans(trainingGroup,k, 'Replicates',225);
%% 3.3 Classification

for i = 1:k
    %get the most occurring division in cluster i
    centroidLabels(i) = mode(cntyShuffled(idx2 == i));
end

clusterAssignments = 1:size(testingGroup,1);

for i = 1:size(testingGroup,1)
    smallestDistance = Inf;
    clusterAssignment = centroidLabels(randi(k));
    for j = 1:k
        distance = norm(testingGroup(i,:) - centroids(j,:));
        if distance < smallestDistance
            smallestDistance = distance;
            clusterAssignment = centroidLabels(j);
        end
    end
    clusterAssignments(i) = clusterAssignment;
end

%% 3.3.1 Training & Testing
%For each test data point, find which centroid it is closest to (nearest
%neighbor), then grab the classification for that centroid. To determine
%accuracy, use cntyShuffled and index to find the actual division for the
%test data, use math to calculate accurancy percentage
testingDivisions = cntyShuffled(181:225);
clusterAssignments = clusterAssignments';
correctDivision = 0;

for i = length(clusterAssignments)
    if testingDivisions(i) == clusterAssignments(i)
        correctDivision = correctDivision + 1;
    end
end

correctDivision / length(clusterAssignments)
