% Copy of Yan emis code, but with a couple changes so I can easily
% run it locally on longwave to test.

fname_srf = '/data/rttools/TIRS_ancillary/PREFIRE_SRF_v0.09.2_2020-02-21.nc';
wn_min = ncread(fname_srf,'channel_wavenum1');
wn_max = ncread(fname_srf,'channel_wavenum2');
wn_tot = cat(2,wn_min,wn_max);
valid_emis = [10 12:16 20:27];   % selected PREFIRE channels
wn_valid = wn_tot(valid_emis,:);
% select out the index of non-valid PREFIRE channels within the
% spectral range
temp = valid_emis(1) : valid_emis(end);
nonvalid_emis = temp(~ismember(temp,valid_emis));
wn_nonvalid = wn_tot(nonvalid_emis,:);

wn_new = load('PCRTM_mono_f.txt');
%wn_new; % spectral grids of input to PCRTM (sensor id: 2), size:
        % 740*1
emis0 = 1:14; % surface emissivity values at 14 selected PREFIRE channels
emis_new = zeros(740,1); % surface emissivity input to the PCRTM

% if the wavenumber falls in any PREFIRE channel
for i = 1:length(valid_emis)
   idw = find(wn_new >= wn_valid(i,1) & wn_new <= wn_valid(i,2));
   emis_new(idw,1) = emis0(i);
end

% if the wavenumber falls beyond the PREFIRE spectral range
id_less = find(wn_new < min(wn_valid(:,1)));
id_larger = find(wn_new > max(wn_valid(:,2)));
emis_new(id_less,1) = emis_new(id_less(end)+1,1);
emis_new(id_larger,1) = emis_new(id_larger(1)-1,1);

% make copies for the alternate methods, since they only
% differ on how they deal with the nonvalid channels in the gaps
% between valid channels.
emis_new2 = emis_new;
emis_new3 = emis_new;

% if the wavenumber does not falls in any valid(i.e. selected)
% PREFIRE channel 
% but still within the spectral range
for j = 1:length(nonvalid_emis)
   idw = find(wn_new >= wn_nonvalid(j,1) & wn_new <= wn_nonvalid(j,2));
   if ismember(nonvalid_emis(j)-1,valid_emis)
       if ismember(nonvalid_emis(j)+1, valid_emis)           
           emis_new(idw,1) = ( emis0( nonvalid_emis(j)-valid_emis(1) ) ...
               + emis0( nonvalid_emis(j)-valid_emis(1)+1 ) )/2;
       else
           emis_new(idw,1) = emis0(nonvalid_emis(j)-valid_emis(1));
       end   
   else
       if ismember(nonvalid_emis(j)+1, valid_emis)
           emis_new(idw,1) = emis0( nonvalid_emis(j)-valid_emis(1)+1 );
       else
           emis_new(idw,1) = emis0(nonvalid_emis(j)-valid_emis(1));
       end    
   end
       
end

% if the wavenumber does not falls in any valid(i.e. selected)
% PREFIRE channel 
% but still within the spectral range
for j = 1:length(nonvalid_emis)
    % locate nearest channels by index number
    channel_num_absdiff = abs(nonvalid_emis(j) - valid_emis);
    min_absdiff = min(channel_num_absdiff);
    closest_valid_channels = find(channel_num_absdiff == min_absdiff);
    % if there are two closest channels, take the average.
    % otherwise, copy the closest channel.
    idw = find(wn_new >= wn_nonvalid(j,1) & wn_new <= wn_nonvalid(j,2));
    if length(closest_valid_channels) == 1
        emis_new2(idw,1) = emis0(closest_valid_channels(1));
    else
        emis_new2(idw,1) = mean(emis0(closest_valid_channels));
    end
end

% if the wavenumber does not falls in any valid(i.e. selected)
% PREFIRE channel 
% but still within the spectral range
for j = 1:length(nonvalid_emis)
   idw = find(wn_new >= wn_nonvalid(j,1) & wn_new <= wn_nonvalid(j,2));
   if ismember(nonvalid_emis(j)-1,valid_emis)
       if ismember(nonvalid_emis(j)+1, valid_emis)           
           emis_new3(idw,1) = ( emis0( nonvalid_emis(j)-valid_emis(1) -(j-1) ) ...
               + emis0( nonvalid_emis(j)-valid_emis(1)+1 -(j-1)) )/2;
       else
           emis_new3(idw,1) = emis0(nonvalid_emis(j)-valid_emis(1)-(j-1));
       end   
   else
       if ismember(nonvalid_emis(j)+1, valid_emis)
           emis_new3(idw,1) = emis0( nonvalid_emis(j)-valid_emis(1)+1 -(j-1));
       else
           emis_new3(idw,1) = emis0(nonvalid_emis(j)-valid_emis(1)-(j-1));
       end    
   end
       
end


figure();
subplot(3,1,1)
channel_wn = mean(wn_tot, 2);
plot(channel_wn(valid_emis), emis0, 'o');
title('Emissivity at valid channels');
axis([250, 1500, 0, 15])
subplot(3,1,2)
plot(wn_new, emis_new)
title('Emissivity at Mono freq with original method')
axis([250, 1500, 0, 15])
subplot(3,1,3)
plot(wn_new, emis_new3)
title('Emissivity at Mono freq with new method')
xlabel('Wavenumber [1/cm]')
axis([250, 1500, 0, 15])

figure()
plot(wn_new, emis_new)
hold on
plot(wn_new, emis_new2)
plot(wn_new, emis_new3)
hold off
legend({'original', 'method 2', 'method 3'})
xlabel('Wavenumber [1/cm]')
axis([250, 1500, 0, 15])
