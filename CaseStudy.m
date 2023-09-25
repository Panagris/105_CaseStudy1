close all;
load("COVIDbyCounty.mat");

jax = max(CNTY_CENSUS(CNTY_CENSUS.REGION == 1,:).POPESTIMATE2021);

[idx, C] = kmeans(CNTY_COVID',9);

figure;
axis equal;
hold on;
for i = 1:9
    plot(CNTY_COVID(1,idx==i), CNTY_COVID(2,idx==i), '.', 'MarkerSize',10);
end
plot(C(:,1),C(:,2),'kx','MarkerSize',14,'LineWidth',3);
hold off;