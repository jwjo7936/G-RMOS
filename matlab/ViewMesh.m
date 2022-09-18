function p = ViewMesh(ptVec,trgVec,color,camera_pos)

if nargin<3 || isempty(color)
%     color =[18 219 200]./255; 
     color = [1.0 0.78 0.67];
end;

%figure
p = patch('Vertices',ptVec,'Faces',trgVec);
set(p,'FaceColor',color);
%set(p,'FaceVertexCData',color,'FaceColor','interp');
set(p,'edgecolor','none');
%set(p,'LineWidth',0.1);
daspect([1 1 1])
%view(3)
if nargin<4
    %view(-84,72);
    %view(-84,-72);
    %view(180,90);
    view(180,-90)
else
    view(camera_pos)
    %campos(camera_pos);
end

% if nargin>4
%     camtarget(camear_target);
% end
% 
% if nargin>5
%     camup(camear_up);
% end

axis tight off
%camlight
lighting gouraud
material dull
colormap(jet)
cameratoolbar
