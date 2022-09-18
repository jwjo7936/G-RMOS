function N = MeshNormal(coord,trg)
% N = MeshNormal(coord,trg)
% Compute the normal direction at each vertex of the mesh.

A = coord(trg(:,2),:) - coord(trg(:,1),:);
B = coord(trg(:,3),:) - coord(trg(:,1),:);
N = cross(A,B);
N2 = zeros(size(coord));
for i=1:size(trg,1)
    N2(trg(i,:),:) = N2(trg(i,:),:) + repmat(N(i,:),[3 1]);
end;
