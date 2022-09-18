function writeObjMesh2(coord,normal,face,fname,texture)
% writeObjMesh2(coord,normal,face,fname,texture)
% write a triangular mesh into an obj file
% Inputs: 
%   coord: 3d coordinates of vertices
%   normal: normal vector at each vertex
%   face: the triangular structure
%    fname: the name of the minc obj format. Need to be "xxx.obj"
%   texture: optional
%
% Yonggang Shi
% Lab of Neuro Imaging (LONI), UCLA School of Medicine. 

face = face - 1; % to convert the index to start from zero

num_face = size(face,1);
num_vert = size(coord,1);
fid = fopen(fname,'w');
%fprintf(fid,'%s ','P');
fprintf(fid,'P 0.3 0.7 0.5 100 1 ');

fprintf(fid,'%d \n',num_vert);

for i=1:num_vert
    fprintf(fid,'%7.4f ',coord(i,:));
    fprintf(fid,'\n');
end;

%normals at each points
for i=1:num_vert
    fprintf(fid,'%5.2f ',normal(i,:));
    fprintf(fid,'\n');
end;
fprintf(fid,'\n');
%number of triangles
fprintf(fid,'%d \n',num_face);

fprintf(fid,'%d ',2);
% pt_colors = [0.0    0.5    0.75 1];
%pt_colors = [1.0 0.78 0.67 1];
pt_colors = [0.7 0.6 0.3 1];
for i=1:num_vert
    if nargin==5
        pt_colors(1:3) = texture(i,:);
    end;
    fprintf(fid,'%4.2f ', pt_colors);
    fprintf(fid,'\n');
end;
fprintf(fid,'\n');

count = 0;
for i=1:num_face
    fprintf(fid,'%d ',3*i);
    count = count + 1;
    if count==8
        fprintf(fid,'\n');
        count = 0;
    end;
end;

fprintf(fid,'\n');

count = 0;
for i=1:num_face
    for j=1:3
        fprintf(fid,'%d ',face(i,j));
        if count==8
             fprintf(fid,'\n');
             count = 0;
        end;
    end;
end;

fclose(fid);

