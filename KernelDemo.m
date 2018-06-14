s=10;
u = [5; 5];
xr = u(1)+2*s;
yr = u(2)+2*s;
[X Y] = meshgrid(-xr:xr,-yr:yr);
Z=exp(-1/s^2*((X-u(1)).^2+(Y-u(2)).^2));
XM = X;
YM = Y;
X = X(:);
Y = Y(:);
Z = Z(:);
XY = [X';Y'];

R = vecnorm(XY-u);
r = 7*s;
mask = R > sqrt(r);
mask1 = R < sqrt(r);


X1 = X(mask);
Y1 = Y(mask);
Z1 = Z(mask);
X2 = X(mask1);
Y2 = Y(mask1);
Z2 = Z(mask1);
figure
plot3(X1,Y1,Z1,'rx')
hold on 
plot3(X2,Y2,Z2,'bx')
Z3 = 0.5*ones(size(XM));
%surf(XM,YM,Z3)
title('Kernel Trick: Graph of x')
hold off