clear all
close all

z_10=load("baseline_LFP_speed_PLV_10Hz.mat");
z_145=load("baseline_LFP_speed_PLV_145Hz.mat");
 
PLV_10=z_10.aCOHs;
freq=z_10.freq;

PLV_145=z_145.aCOHs;

bs_10=nanmedian(PLV_10);
bs_145=nanmedian(PLV_145);

ranksum(bs_10',bs_145')

figure('COlor','w','Position', [300 400 120 90],'Renderer', 'painters') 
fill_error_area2(freq,nanmedian(PLV_10,2), nanstd(PLV_10,[],2)./sqrt(size(PLV_10,2)),[ 0.5 0.5 0.5])
fill_error_area2(freq,nanmedian(PLV_145,2), nanstd(PLV_145,[],2)./sqrt(size(PLV_145,2)),[ 0.5 0.5 0.5])
plot(freq,nanmedian(PLV_10,2),'b','Linewidth',1)
plot(freq,nanmedian(PLV_145,2),'r','Color', [ 0.8 0.2 .2],'Linewidth',1)
ylabel('LFP-Speed PLV')
xlabel('Frequency (Hz)')
axis tight
xlim([1,10])
xticks([2:2:10])
%ylim([0,0.03])
yticks(0:0.01:0.03)



figure('COlor','w','Position', [300 400 120 90],'Renderer', 'painters') 
plot(bs_10)
hold on
plot(bs_145)