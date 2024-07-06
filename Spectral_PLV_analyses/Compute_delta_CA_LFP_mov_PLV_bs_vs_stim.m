clear all
close all

%% Ca-speed and Ca-LFP PLV 
%%  Load motion and mean fluorescence trace
%cd('/home/hanlabadmins/eng_handata/eng_research_handata2/Sudi_Sridhar/612535/07082020/612535_10Hz_AudioVisual/motion_corrected')
addpath(genpath('U:\eng_research_handata\EricLowet'))
mainroot='U:\eng_research_handata\EricLowet\SUDI';
datapath='U:\eng_research_handata\eng_research_handata2\Sudi_Sridhar\';

aCOHs=[];aSPs=[];aCOHs2=[];aPHs=[];aPHs_SH=[];aNT=[];corrLFP=[];mouse_id=[];bs_vect=[];stim_vect=[];PLV_vect=[];
cha=1;% 2= movement, 1=LFP
for stim_freq=1 % 1= 10Hz, 2=145Hz
    
if stim_freq==1
% 10Hz
allSES{1}='\608448\01062020\608448';  % no LFP
allSES{2}='\608451\01062020\608451';
allSES{3}='\608452\01062020\608452';
allSES{4}='\608450\03092020\608450';
allSES{5}='\602101\03092020\602101';
allSES{6}='\611311\03092020\611311';
allSES{7}='\608448\01172020\608448';
allSES{8}='\608451\07082020\608451';
allSES{9}='\608452\07082020\608452';
allSES{10}='\611111\07082020\611111';
allSES{11}='\602101\07082020\602101';
allSES{12}='\612535\07082020\612535';
allSES{13}='\615883\02052021\10Hz\615883';
allSES{14}='\615883\03122021\10Hz\615883';
else
% 140Hz
allSES{1}='\608448\01102020\608448';
allSES{2}='\608451\01102020\608451';
allSES{3}='\608452\01102020\608452';
allSES{4}='\602101\03112020\602101';
allSES{5}='\611311\03112020\611311';
allSES{6}='\611111\03112020\611111';
allSES{7}='\608450\01172020\608450';
allSES{8}='\608451\07012020\608451';
allSES{9}='\608452\07012020\608452';
allSES{10}='\611111\07012020\611111';
allSES{11}='\612535\07022020\612535';
allSES{12}='\602101\07022020\602101'; %exclude LFP
allSES{13}='\615883\02052021\145Hz\615883';
allSES{14}='\615883\03122021\145Hz\615883';
end

if stim_freq==1
ses_sel=[ 2:13];
else
  ses_sel=[ 1:11 13 14];  
end

%ses_sel=2;
for ses=  ses_sel


if  stim_freq==1
cd([ datapath  allSES{ses} '_10Hz_AudioVisual\motion_corrected']);
else
 cd([ datapath  allSES{ses} '_145Hz_AudioVisual\motion_corrected'])   
end




motion=h5read('processed_motion.h5','/raw_speed_trace');
tracesF=h5read('processed_trace.h5','/trace');
traces=h5read('processed_trace.h5','/onset_binary_trace');
load('LFP_ts.mat')
stim_onsets=LFP_data.Stim_onset;
stim_offsets=LFP_data.Stim_offset;
v=motion;
idx=find(isnan(v)==0);
%Remove NANs
v = v(~isnan(v));  % motion signal
% Align traces and motiontraces=traces(idx,:);
traces=traces(idx,:);tracesF=tracesF(idx,:);

%% Extra -stim vect
Sampling_freq=20;
time_vect1  = 0:1/Sampling_freq:2099.9;% Original time vector (20Hz)
signal1=v ; % signal
time_vect2 = 0:1/1000:2099.9; % time vector for interpolation (1000Hz)
%signal1_Intp=interp1(time_vect1,signal1,time_vect2);%  interpolated signal
stim_vec=zeros(1,size(tracesF,1));
for i=1:length(stim_onsets)
timsel=(stim_onsets(i)-0:stim_onsets(i)+1200-1) -idx(1);
stim_vec(timsel)=1;
end
%stim_vec=interp1(time_vect1,stim_vec(1:length(time_vect1)),time_vect2);%  interpolated signal


% resp_cells=h5read('Motion_resp_cells_sustained_10Hz.h5',allSES{4});
% resp_cells=squeeze(resp_cells);
% % Python to MATLAB
% resp_cells=resp_cells+1;
% 
% zscore(lfp_all.trial{1}(1,:))

Sampling_freq=20;
% Mean across all neuron
mean_traces=zscore(nanmean(traces,2));  %% Here is use median instead of mean as a better population average
% High-speed periods
moving_period=h5read('processed_motion.h5','/moving_period');
moving_period=moving_period(idx);  % moving period in 0 and 1

LFP=LFP_data.LFP;
%asel=fastsmooth(zscore(fastsmooth(abs(hilbert(LFP)),5,1,1))>4,300,1,1);
%LFP(asel>0)=median(LFP)+randn(1, length(find(asel>0))).*std(LFP);
%figure(1)
%plot(LFP)
 vv=LFP(1:50:end); %down-sampling
 vv=vv(idx);
%% Align LFP and motion
start_frame=LFP_data.Start_Imaging;
stim_onsets=LFP_data.Stim_onset;
stim_offsets=LFP_data.Stim_offset;
delay_frames=start_frame+ idx(1);
delay_frame_LFP = ceil(delay_frames*1000/20);
%aligned_LFP = LFP(delay_frame_LFP:delay_frame_LFP+length(signal1_Intp)-1);
shifted_stim_onsets=(stim_onsets-idx(1))/20*1000;
shifted_stim_offsets=(stim_offsets-idx(1))/20*1000;
aligned_LFP=LFP(delay_frame_LFP:50:end);
aligned_LFP=aligned_LFP(1:length(moving_period));



%%%%%%%%%%%%%%%%%%%%%%%
if 1
 
    lfp_all=[];FS=20
    lfp_all.trial{1}(1,:)= (aligned_LFP);%zLFP
        lfp_all.trial{1}(2,:)= zscore(v-fastsmooth(v,400,1,1)); %Motion signal
    lfp_all.time{1}= (1:size(lfp_all.trial{1},2))./FS;
    lfp_all.label= {'LFP', 'motion'};
      
%%%%%%%%%%%%%%%
deltP=lfp_all.trial{1}(2,:);
Fn = FS/2;FB=[ 3 4 ];
[B, A] = butter(2, [min(FB)/Fn max(FB)/Fn]);
deltP= abs(hilbert(filtfilt(B,A,    deltP)));
thresD= prctile(deltP(moving_period==1),30);   
% thresD is a threshold on movement delta power as additional criterion. The idea is
%that to compute PLV only in time periods when there is suffcient movement delta
%power. 

corrLFP=[corrLFP,xcorr(zscore(v-fastsmooth(v,400,1,1)), aligned_LFP,2500,'Coeff')]
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  cfg = []; %block_type == cfg.blk
% cfg.method ='wavelet';%mtmfft'; %'mvar';
% cfg.output ='fourier';
% cfg.taper='hanning';
% cfg.keeptapers ='yes';
% cfg.keeptrials ='yes';
% cfg.trials='all';cfg.tapsmofrq =5;%
% cfg.channel= ['all']; %chans=cfg.channel;
% cfg.foi= [1:0.5:9.5];    cfg.t_ftimwin =[ones(1,length(cfg.foi))*1];
% cfg.toi= lfp_all.time{1};   cfg.width =6;
% freq2 = ft_freqanalysis(cfg, lfp_all);

[cwt_out,frs]=runcwt(lfp_all, [1 20],FS);

freq2.freq=frs;
wavA = abs(squeeze(cwt_out(1,cha,:,:)));
wavD = angle(squeeze(cwt_out(1,cha,:,:)));

wavD2=wavD;
moving_period=moving_period';
wavD(:,~(deltP>thresD  & (moving_period==1 &stim_vec==0)))=NaN;
wavD2(:,~(deltP>thresD & (moving_period==1 &stim_vec==1)))=NaN;


mm=0;mm3=0;clear allCOHS allC allCOHS2 allSP allPH1 allPH2 allNT allNT2
for neuron=1:size(tracesF,2)
 [ s_oasis, onsets2 ,onsets]= deconvSudi(tracesF(:,neuron));
minimum_spikes=20;
% Can try using only wavD2
    if sum(~isnan(wavD2(1,onsets2))) >minimum_spikes %& sum(~isnan(wavD2(1,onsets2))) >minimum_spikes
        mm=mm+1
        
        %%%%%%%% PLV %%%%%%%%%%%%%%
        T=    abs(nanmean(exp(1i.*wavD(:,onsets2)),2)).^2;
        NT=sum(~isnan(wavD(1,onsets2)));
        allCOHS(:,mm)=    (((1/(NT-1))*((T.*NT-1)))) ;% abs(nanmean(exp(1i.*wavD(:,onsets2)),2));
        %%%%%%%%% PHASE %%%%%%%%%%%%%%%%%%%%%%%
        allPH1(:,mm)=    angle(nanmean(exp(1i.*wavD(:,onsets2)),2));
        allNT(mm)=NT;

        NT2=sum(~isnan(wavD2(1,onsets2)));
        mm3=mm3+1;
        T=    abs(nanmean(exp(1i.*wavD2(:,onsets2)),2)).^2;
        allCOHS2(:,mm3)=    (((1/(NT2-1))*((T.*NT2-1)))); 
        allPH2(:,mm)=    angle(nanmean(exp(1i.*wavD2(:,onsets2)),2));
        
 clear dataT2
mm2=0;wind=40;
for id=1:length(onsets2)
    if onsets2(id)>wind  & onsets2(id)+wind <size(traces,1) 
        mm2=mm2+1;
   dataT2(:,mm2)= zscore(tracesF(onsets2(id)-wind:  onsets2(id)+wind,neuron  ));  
    end
    
end
allSP(:,mm)=nanmean(dataT2,2);
    end
end
end

if mm>0
    Dx=max(diff(allSP));

calc_transient_perc=0;
aCOHs= [aCOHs, allCOHS(:,Dx>prctile(Dx,calc_transient_perc))];
aCOHs2= [aCOHs2, allCOHS2(:,Dx>prctile(Dx,calc_transient_perc))];
aPHs= [aPHs, allPH1(:,Dx>prctile(Dx,calc_transient_perc))];
aPHs_SH= [aPHs_SH, allPH2(:,Dx>prctile(Dx,calc_transient_perc))];
aSPs= [aSPs, allSP(:,Dx>prctile(Dx,calc_transient_perc))];
aNT=[aNT, allNT(Dx>prctile(Dx,calc_transient_perc))];

mouse_id_sess= ses *ones(1,size(allCOHS(:,Dx>prctile(Dx,calc_transient_perc)),2));
mouse_id=[mouse_id,mouse_id_sess];

bs_vect=[bs_vect,zeros(1,size(allCOHS(:,Dx>prctile(Dx,calc_transient_perc)),2))];
stim_vect=[stim_vect,ones(1,size(allCOHS(:,Dx>prctile(Dx,calc_transient_perc)),2))];

end

end
end
% varNames = {'MouseID','Stim','PLV'};
% T=table(mouse_id',stim_vect',PLV_vect','VariableNames',varNames);
% %writetable(T,'C:\Users\sudiksha\Documents\Codes\Ca_LFP_PLV_10Hz.csv');
% %T=readtable('C:\Users\sudiksha\Documents\Codes\Ca_LFP_PLV_10Hz.csv','ReadVariableNames',true);
% glme = fitglme(T,'PLV~ 1+Stim+(1|MouseID)',...
%     'Distribution','Gaussian','Link','identity');
% disp(glme)

savepath='C:\Users\sudiksha\Documents\Codes\Spectral_analysis\Figures\'
nt=20;
figure('COlor','w','Position', [ 300 400 250 180],'Renderer', 'painters') 
fill_error_area2(freq2.freq,nanmean(aCOHs(:,aNT>nt),2), nanstd(aCOHs(:,aNT>nt),[],2)./sqrt(size(aCOHs(:,aNT>nt),2)),[ 0.5 0.5 0.5])
fill_error_area2(freq2.freq,nanmean(aCOHs2(:,aNT>nt),2), nanstd(aCOHs2(:,aNT>nt),[],2)./sqrt(size(aCOHs2(:,aNT>nt),2)),[ 0.5 0.5 0.5])
plot(freq2.freq,nanmean(aCOHs(:,aNT>nt),2),'blue')
hold on
plot(freq2.freq,nanmean(aCOHs2(:,aNT>nt),2),'red')
axis tight
%print(gcf, '-dpdf' , '-r300' ,'-painters', [ savepath 'Ca_LFP_PLV_lock_bs_10Hzstim' num2str(stim_freq) '.pdf'])

nt=20;
figure('COlor','w','Position', [ 300 400 250 180],'Renderer', 'painters') 
fill_error_area2(freq2.freq,nanmean(aCOHs(:,aNT>nt),2), nanstd(aCOHs(:,aNT>nt),[],2)./sqrt(size(aCOHs(:,aNT>nt),2)),[ 0.5 0.5 0.5])
plot(freq2.freq,nanmean(aCOHs(:,aNT>nt),2),'blue')
axis tight

nt=20;
figure('COlor','w','Position', [ 300 400 250 180],'Renderer', 'painters') 
fill_error_area2(freq2.freq,nanmean(aCOHs2(:,aNT>nt),2), nanstd(aCOHs2(:,aNT>nt),[],2)./sqrt(size(aCOHs2(:,aNT>nt),2)),[ 0.5 0.5 0.5])
plot(freq2.freq,nanmean(aCOHs2(:,aNT>nt),2),'red')
axis tight

% Mixed effect models here?
frsel=find(freq2.freq>2 & freq2.freq<4) 
%frsel=find(freq2.freq>0 & freq2.freq<1.5)  
delt_angs= circ_mean(aPHs(frsel,:));
delt_angs_SH= circ_mean(aPHs_SH(frsel,:));

delt_PLV= nanmean(aCOHs(frsel,:));
delt_PLV_SH= nanmean(aCOHs2(frsel,:));

figure('COlor','w','Position', [ 300 400 120 90],'Renderer', 'painters') 
polarhistogram(delt_angs(delt_PLV>0.0),[-pi:0.33:pi],'FaceColor','blue','Normalization','probability','Facealpha',0.3);hold on,
%print(gcf, '-dpdf' , '-r300' ,'-painters', [ savepath 'Ca_LFP_PLV_lock_bs_10Hz' num2str(stim_freq) '.pdf'])

figure('COlor','w','Position', [ 300 400 120 90],'Renderer', 'painters') 
polarhistogram(delt_angs_SH(delt_PLV_SH>0.0),[-pi:0.33:pi],'FaceColor','red','Normalization','probability','Facealpha',0.3);hold on,
%print(gcf, '-dpdf' , '-r300' ,'-painters', [ savepath 'Ca_LFP_PLV_lock_10Hzstim' num2str(stim_freq) '.pdf'])

%% Stats for preferred phase
pref_phase_bs= delt_angs(delt_PLV>0.0);
pref_phase_stim= delt_angs_SH(delt_PLV_SH>0.0);
p=circ_cmtest(pref_phase_bs,pref_phase_stim);
C = {pref_phase_bs; pref_phase_stim};
[bH, fPEst, fWTest, strPMethod]= mardiatestn_circ_equal(C)

% pref_phase=[pref_phase_bs,pref_phase_stim];
% mouse_id2=mouse_id;
% mouse_id = mouse_id(delt_PLV>0.0);
% mouse_id2 = mouse_id2(delt_PLV_SH>0.0);
% mouse_id_tot=[mouse_id,mouse_id2];
% 
% bs_vect = bs_vect(delt_PLV>0.0);
% stim_vect = stim_vect(delt_PLV_SH>0.0);
% stim_vect=[bs_vect,stim_vect];
% 
% %% Change this to circular stats 
% 
% varNames = {'MouseID','Stim','Phase'};
% T=table(mouse_id_tot',stim_vect',pref_phase','VariableNames',varNames);
% glme = fitglme(T,'Phase~ 1+Stim+(1|MouseID)',...
%      'Distribution','Gaussian','Link','identity');
% disp(glme)
