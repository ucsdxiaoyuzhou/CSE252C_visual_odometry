close all;
clear all;
pose = load('pose08.txt');
[dataNum, ~] = size(pose);

originalPt = [0;0;0;1];
tempTrans = eye(4);
position = originalPt;
for i = dataNum :-1: 1
% for i = 1 : dataNum
    temp = pose(i, 1:3);
    temp = rotationVectorToMatrix(temp);
    tempTrans(1:3,1:3) = temp;
    tempTrans(1:3,4) = pose(i, 4:6);
   
    tempPosition = tempTrans * position;
    position = [originalPt, tempPosition];
end

% for i = dataNum :-1: 1
%     temp = pose(i, 1:3);
%     temp = rotationVectorToMatrix(temp);
%     tempTrans(1:3,1:3) = temp;
%     tempTrans(1:3,4) = pose(i, 4:6);
%     
%     position = tempTrans\position;
%     
% end

plot(-position(1,:), -position(3,:));
axis equal
xlabel('x');
ylabel('y');
grid
hold on

%% plot ground truth 
groundtruth = load('08.txt');
[dataNum, ~] = size(groundtruth);

originalPt = [0;0;0;1];
tempTrans = eye(4);
position = originalPt;
for i = 1 : dataNum
    tempTrans(1,:) = groundtruth(i,1:4);
    tempTrans(2,:) = groundtruth(i,5:8);
    tempTrans(3,:) = groundtruth(i,9:12);
    
    position = [position, tempTrans*originalPt];
end
% figure
plot(position(1,:), position(3,:));
axis equal
% axis([minz, maxz, minz, maxz, minz, maxz]);
xlabel('x');
ylabel('y');
grid
legend('our method', 'ground truth');
hold off







