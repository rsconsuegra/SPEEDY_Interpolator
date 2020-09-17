clc
clear all
%fid=fopen('./prev_state/1982010100.grd', 'r','b');
file_name = '1982010706.grd';
%hr1=fread(fid, 1, '*uint32')
fid=fopen(file_name, 'r','b');
nlon=96;
nlat=48;
nlev=7;

for k=1:nlev
    for j =1:nlat
        for i=1:nlon
            U(i,j,k)=fread(fid, 1, 'float32');
        end
    end
end


for k=1:nlev
    for j =1:nlat
        for i=1:nlon
            V(i,j,k)=fread(fid, 1, 'float32');
        end
    end
end

for k=1:nlev
    for j =1:nlat
        for i=1:nlon
            T(i,j,k)=fread(fid, 1, 'float32');
        end
    end
end

for k=1:nlev
    for j =1:nlat
        for i=1:nlon
            Q(i,j,k)=fread(fid, 1, 'float32');
        end
    end
end

for j =1:nlat
    for i=1:nlon
        PS(i,j)=fread(fid, 1, 'float32');
    end
end

fclose(fid);