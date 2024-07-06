clear all
close all

z_10=load("baseline_speed_power_10Hz.mat");
z_145=load("baseline_speed_power_145Hz.mat");
 
power=z_10.p_all_sess4;
freq=z_10.freq;

power_145=z_145.p_all_sess;

bs_10=nanmean(power);
bs_145=nanmean(power_145);

ranksum(bs_10',bs_145')

figure('COlor','w','Position', [300 400 120 90],'Renderer', 'painters') 
fill_error_area2(freq,nanmean(power,2), nanstd(power,[],2)./sqrt(size(power,2)),[ 0.5 0.5 0.5])
fill_error_area2(freq,nanmean(power_145,2), nanstd(power_145,[],2)./sqrt(size(power_145,2)),[ 0.5 0.5 0.5])
plot(freq,nanmean(power,2),'b','Linewidth',1)
plot(freq,nanmean(power_145,2),'r','Color', [ 0.8 0.2 .2],'Linewidth',1)
ylabel('Normalized PSD of speed')
xlabel('Frequency (Hz)')
axis tight
xlim([0,10])
ylim([0,12])
xticks(2:2:10)



