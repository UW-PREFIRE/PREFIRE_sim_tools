% Data format in 12-hourly simulations (ieee file)
%     Each ieee file contains 121*240 records (or ECMWF ERA 1.5-deg by 1.5-deg grids).
%     Each record contains (1+1+400+400) float values. They are 1 latitude, 1 longitude, 
%     400 all-sky PC scores, 400 clear-sky PC scores, respectively.

% This code is to read PC scores in ieee files, 
% then convert the PC scores to radiances (W per m^2 per sr per cm^-1) in a spectral interval of 0.5 cm-1.

clear all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% configuration %%%%%%%%%%%%%%%%%%


% path sotring the PC score data
fpath = '../data/';

% For instruction, refer to Instruction_for_PCRTM_installation_v3.4.doc in PCRTM_V3.4
sensor_id = 2;  % CLARREO 0.5 cm-1 spectral resolution


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %  b. PC components stored in PCRTM package

fid=fopen(['Pcnew_id',int2str(sensor_id),'.dat'],'r','l');  

% data structure   
% num of bands, number of PCs in each band, and number of channels in each band
% these are all fixed value, reading them just for double-check
numbnd		= fread(fid, 1, 'float'); % number of bands
numPC 		= fread(fid, numbnd, 'float');   % number of PCs in each band
                % 100 PC in each band
numch		= fread(fid, numbnd, 'float');        
						
     
PCcoef = zeros(numbnd,100,max(numch))+NaN; % Each band has 100 PC loadings, 
				 % each loading has a dimension of numch  
Pstd = zeros(numbnd,max(numch))+NaN;

Pmean = zeros(numbnd,max(numch))+NaN;

IDX = cell(numbnd, 1);
numch0 = [1, numch'];

for i = 1:numbnd
       IDX{i} = 1:numch(i);
       if i >1
         IDX{i} = IDX{i} + sum(numch(1:i-1));
       end
end
nn =0;
for ib =1:numbnd  
    for ip =1:numPC(ib)
        nn = nn + numch(ib);
	PCcoef(ib, ip, 1:numch(ib)) = reshape(fread(fid, numch(ib), 'float'), 1, 1, numch(ib));
    end
    	Pstd(ib,1:numch(ib))= reshape(fread(fid, numch(ib), 'float'), 1, numch(ib));
        
     Pmean(ib,1:numch(ib))= reshape(fread(fid, numch(ib), 'float'), 1, numch(ib));
         
end

wn_PCRTM = fread(fid, sum(numch), 'float'); 

fclose(fid);

nPC = sum(numPC);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
nlat = 6144;

nlon = 12288;

UTCstr={'03'};

    

   %%%%%
   %  read simulated data and convert PC scores to radiances
 iyear = 2016;
 for imon =8
 
                 
   for iday =1

        
     for it=1
   
         
        for ilat=1:nlat
           fname=[fpath,'PC_GFDL_',int2str(iyear),myint2str(imon,2),myint2str(iday,2),'_',UTCstr{it},'UTC_ilat',myint2str(ilat,4),'.par_spec.ieee'];
            
           if exist(fname,'file')
            
             fid = fopen(fname, 'rb','l');
           
             fseek(fid, 0, 1);
	         reclen = ftell(fid)/(4*2+4*nPC);

	         fseek(fid, 0, -1); 
 
             for ilon = 1:reclen
                 
	                           
                        tmpout = fread(fid, 2 , 'float');
                        %tmpout  contains latitude and longitude
                     
                        
                        tmp_PC0 = fread(fid, nPC , 'float'); 
                        
                        lat2(ilat) = tmpout(1);
                        lon2(ilon) = tmpout(2);
                  
                        tmp_PC =  reshape(tmp_PC0,numPC(1),numbnd); 
                        if isnan(tmp_PC0(1)) | abs(tmp_PC0(1))>1e5 | isinf(abs(tmp_PC0(1)))
                            disp('wrong PCs');
                            [ilat ilon]
                        else 
                            for ib =1:numbnd   
                           
                               aa = tmp_PC(:, ib)' * reshape(PCcoef(ib, :, 1:numch(ib)), numPC(ib), numch(ib));
		  
                               rad(IDX{ib}) = ((aa +1) .* Pstd(ib,1:numch(ib)))' *1e-3; % W per m^2 per sr per cm^-1

                               clear aa;
                            end  % end of bands  


                        end
                         

                                            
              end  % end of record   
              fclose(fid);
               
         else
             disp([fname,' not exist']);
         end
        end
         
    end  % end of day


  end
end


