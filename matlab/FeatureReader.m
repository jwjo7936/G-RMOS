function pt_colors = FeatureReader(fname) 

fid = fopen(fname, 'r');

pt_colors = fread(fid, 1000,'float')
pt_colors = reshape(pt_colors, 1, 1000);
pt_colors = pt_colors';
        
fclose(fid);
