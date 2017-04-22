pose = load('pose08.txt');
[dataNum, ~] = size(pose);

originalPt = [0;0;0;1];
tempTrans = eye(4);
position = [];
for i = 1 : dataNum
    temp = pose(i, :);
    tempTrans(1,:) = temp(1:4);
    tempTrans(2,:) = temp(5:8);
    tempTrans(3,:) = temp(9:12);
    
    position = [position, tempTrans*originalPt];
end
position = position(1:3,:);
maxz = max(position(3,:));
minz = min(position(3,:));

maxx = max(position(1,:));
minx = min(position(1,:));

plot3(position(1,:), position(2,:), position(3,:), 'o');
% axis([minz, maxz, minz, maxz, minz, maxz]);
xlabel('x');
ylabel('y');
zlabel('z');
grid
