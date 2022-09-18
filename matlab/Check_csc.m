fid = fopen('data/ASparseMtxE_Y.raw', 'r');

data = fread(fid, 1000000,'double');
data = reshape(data, 1000, 1000);

x = diag(data)

fclose(fid);

disp(isequal(data,data.'))