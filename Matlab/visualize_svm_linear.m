function visualize_svm_linear(x1, x2, B, B0, SV, t, z, c)
%VISUALIZE_SVM_LINEAR Summary of this function goes here
%   Detailed explanation goes here
% Linearly Seperable

    x = [x1 x2];
    xt = min([x1(1,:) x2(1,:)]):.25:max([x1(1,:) x2(1,:)]);

    h = gcf;
    % Plot data points from classes
    plot(x1(1,:),x1(2,:),'rx')
    hold on
    plot(x2(1,:),x2(2,:),'bx')

    % Boundary..
    h = -(B(1)/B(2))*xt - B0/B(2);
    hl = -(B(1)/B(2))*xt - (B0+1)/B(2);
    hh = -(B(1)/B(2))*xt - (B0-1)/B(2);
    plot(xt,h);

    % Support Vectors
    plot(SV(1,:),SV(2,:),'gO')
    % Non-Negative Zeta Vector
    if exist('z','var')
        plot(x(1,z>1e-10),x(2,z>1e-10),'mo')
        % Margins
        plot(xt,hl);
        plot(xt,hh);
        legend('Class 1', 'Class 2', 'Linear Discriminant', 'Support Vectors', 'Points Inside Margin')
        title(t);

    else
        plot(xt,hl);
        plot(xt,hh);
        legend('Class 1', 'Class 2', 'Linear Discriminant', 'Support Vectors', 'Points Inside Margin')   
        title(t);
    end

    hold off

end

