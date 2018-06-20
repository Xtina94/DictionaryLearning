clear;
close all
%% load data
input1 = 'uber/uber_aug14_zone.csv';
fid = fopen(input1);
data1 = textscan(fid,'%s %f %f %*s %s','HeaderLines',1,'Delimiter',',');
fclose(fid);

input2 = 'uber/uber_sep14_zone.csv';
fid = fopen(input2);
data2 = textscan(fid,'%s %f %f %*s %s','HeaderLines',1,'Delimiter',',');
fclose(fid);

%% calculate hourly bin
date_num1 = cellfun(@(x) datenum(x),data1{1});
date_num2 = cellfun(@(x) datenum(x),data2{1});

a = datenum('2014-08-01 00:00:00');
b = datenum('2014-09-01 00:00:00');
c = datenum('2014-10-01 00:00:00');
hours1 = linspace(a,b,24*31+1)';
hours2 = linspace(b,c,24*30+1)';

hourbin1 = arrayfun(@(x) find(x-hours1<0,1)-1,date_num1);
hourbin2 = arrayfun(@(x) find(x-hours2<0,1)-1,date_num2);

%% calculate zone bin
zone_list = unique(data1{4});
[~,zonebin1] = ismember(data1{4},zone_list);
[~,zonebin2] = ismember(data2{4},zone_list);

%% calculate histogram over hourly and zone bins
hist1 = accumarray([hourbin1 zonebin1],1);
hist2 = accumarray([hourbin2 zonebin2],1); hist2 = [hist2;zeros(1,30)];

%% compute the average lat/log coordinates 
for i = 2 : numel(zone_list)
    
    [rn, ~] = find(strcmp(data2{4},zone_list(i)));
    lat2(i-1,1) = mean(data2{2}(rn));
    lon2(i-1,1) = mean(data2{3}(rn));
    
end

for i = 2 : numel(zone_list)
    
    [rn, ~] = find(strcmp(data1{4},zone_list(i)));
    lat1(i-1,1) = mean(data1{2}(rn));
    lon1(i-1,1) = mean(data1{3}(rn));
    
end

lat = mean([lat1 lat2], 2);

lon = mean([lon1 lon2], 2);

%% distinguish weekends from weekdays in September

all_days = hist2(:,2:end)';
weekends = all_days(:,[24*5+1:24*7 24*12+1:24*14 24*19+1:24*21 24*26+1:24*28]);
diff = setdiff([1:720],[24*5+1:24*7 24*12+1:24*14 24*19+1:24*21 24*26+1:24*28]);
weekdays = all_days(:, diff);

