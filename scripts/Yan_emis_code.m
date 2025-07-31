fname_srf = 'PREFIRE_SRF_v0.09.2_2020-02-21.nc';
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

wn_new; % spectral grids of input to PCRTM (sensor id: 2), size:
        % 740*1
emis0; % surface emissivity values at 14 selected PREFIRE channels
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

% if the wavenumber does not falls in any valid(i.e. selected)
% PREFIRE channel 
% but still within the spectral range
for j = 1:length(nonvalid_emis)
   idw = find(wn_new >= wn_nonvalid(j,1) & wn_new <= wn_nonvalid(j,2));
   if ismember(nonvalid_emis(j)-1,valid_emis)
       if ismember(nonvalid_emis(j)+1, valid_emis)           
           emis_new(idw,1) = ( emis0( nonvalid_emis(j)-valid_emis(1)-(j-1) ) ...
               + emis0( nonvalid_emis(j)-valid_emis(1)+1-(j-1) ) )/2;
       else
           emis_new(idw,1) = emis0(nonvalid_emis(j)-valid_emis(1)-(j-1));
       end   
   else
       if ismember(nonvalid_emis(j)+1, valid_emis)
           emis_new(idw,1) = emis0( nonvalid_emis(j)-valid_emis(1)+1-(j-1));
       else
           emis_new(idw,1) = emis0(nonvalid_emis(j)-valid_emis(1)-(j-1));
       end
   end
       
end
