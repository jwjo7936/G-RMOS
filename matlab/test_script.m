[c,n,t] = ReadObjShape('data/target_hippo.obj');
color = FeatureReader('data/target_hippo_mc.raw');

figure(1);
h = ViewMesh(c,t);
%campos(cp), camup(cu), camtarget(ct)
%cam1 = camlight('highlight'); lighting gouraud, material dull

set(h,'EdgeColor','k','EdgeAlpha',0.5)

w = sum((c-repmat(c(1,:),length(c),1)).^2,2); % example feature: distance from the fist node

set(h,'FaceVertexCData',color,'FaceColor','interp')
