folder = '/home/shuoy/nv_work/unitree_ws/src/tightly-coupled-visual-inertial-leg-odometry/output/dataset2_0908/';
% name convention
%  vio - vins-fushion
%  vilo - wob      vilo no bias correction
%  vilo - wb       vilo with bias correction
%  vilo - wb2      vilo with baseline bias correction

% read vio, vilo wob,  vilo wb,
vio_file = strcat(folder,'vio_0908_forward_stop');
vio_wob_file = strcat(folder,'vilo_0908_forward_stop_redix_no_bias_correction');
vio_wb_file = strcat(folder,'vilo2021-09-10_16-42-26');

% read table 
vio_Tab = readtable(vio_file);
vio_wob_Tab = readtable(vio_wob_file);
vio_wb_Tab = readtable(vio_wb_file);
data_start_idx = 1;
data_end_idx = 1;

% first read time, find the one with minimum range 
vio_time = (vio_Tab.Var1-vio_Tab.Var1(1))/10^9;
vio_wob_time = (vio_wob_Tab.Var1-vio_wob_Tab.Var1(1))/10^9;
vio_wb_time = (vio_wb_Tab.Var1-vio_wb_Tab.Var1(1))/10^9;

times = {vio_time,vio_wob_time,vio_wb_time};
end_time = [vio_time(end) vio_wob_time(end) vio_wb_time(end)];
[M,I] = min(end_time);

ref_time = times{I};

% read groundtruth position
gt_pos_x = interp1(vio_wb_time,vio_wb_Tab.Var12,ref_time);
gt_pos_y = interp1(vio_wb_time,vio_wb_Tab.Var13,ref_time);
gt_pos_z = interp1(vio_wb_time,vio_wb_Tab.Var14,ref_time);

%calculate ground truth velocity
gt_vel_x = (gt_pos_x(2:end) - gt_pos_x(1:end-1))./(ref_time(2:end)-ref_time(1:end-1));
gt_vel_y = (gt_pos_y(2:end) - gt_pos_y(1:end-1))./(ref_time(2:end)-ref_time(1:end-1));
gt_vel_z = (gt_pos_z(2:end) - gt_pos_z(1:end-1))./(ref_time(2:end)-ref_time(1:end-1));
% smooth the velocity
gt_vel_x = movmean(gt_vel_x,2,1);
gt_vel_y = movmean(gt_vel_y,2,1);
gt_vel_z = movmean(gt_vel_z,2,1);
gt_vel_x = [0; gt_vel_x(1:end-1)];
gt_vel_y = [0; gt_vel_y(1:end-1)];
gt_vel_z = [0; gt_vel_z(1:end-1)];

% lo velocity

lo_vel_x = interp1(vio_wb_time,vio_wb_Tab.Var18,ref_time);
lo_vel_y = interp1(vio_wb_time,vio_wb_Tab.Var19,ref_time);
lo_vel_z = interp1(vio_wb_time,vio_wb_Tab.Var20,ref_time);


%% read vio pos
% interpolate time to get all pos
vio_pos_x = interp1(vio_time,vio_Tab.Var2,ref_time);
vio_pos_y = interp1(vio_time,vio_Tab.Var3,ref_time);
vio_pos_z = interp1(vio_time,vio_Tab.Var4,ref_time);
% move data to align with ground truth
vio_pos_x = vio_pos_x + gt_pos_x(1) - vio_pos_x(1);
vio_pos_y = vio_pos_y + gt_pos_y(1) - vio_pos_y(1);
vio_pos_z = vio_pos_z + gt_pos_z(1) - vio_pos_z(1);
% rotate data
angle = -1.9/180*pi;
R = [cos(angle)   -sin(angle)  0;
    sin(angle)    cos(angle)  0;
    0                   0     1];
rotated = R * [vio_pos_x';vio_pos_y';vio_pos_z'];
vio_pos_x = rotated(1,:)';
vio_pos_y = rotated(2,:)';
vio_pos_z = rotated(3,:)';

%calculate vio velocity
vio_vel_x = (vio_pos_x(2:end) - vio_pos_x(1:end-1))./(ref_time(2:end)-ref_time(1:end-1));
vio_vel_y = (vio_pos_y(2:end) - vio_pos_y(1:end-1))./(ref_time(2:end)-ref_time(1:end-1));
vio_vel_z = (vio_pos_z(2:end) - vio_pos_z(1:end-1))./(ref_time(2:end)-ref_time(1:end-1));
% smooth the velocity
vio_vel_x = movmean(vio_vel_x,2,1);
vio_vel_y = movmean(vio_vel_y,2,1);
vio_vel_z = movmean(vio_vel_z,2,1);


%% read vilo wob pos
% interpolate time to get all pos
vio_wob_pos_x = interp1(vio_wob_time,vio_wob_Tab.Var2,ref_time);
vio_wob_pos_y = interp1(vio_wob_time,vio_wob_Tab.Var3,ref_time);
vio_wob_pos_z = interp1(vio_wob_time,vio_wob_Tab.Var4,ref_time);
% move data to align with ground truth
vio_wob_pos_x = vio_wob_pos_x + gt_pos_x(1) - vio_wob_pos_x(1);
vio_wob_pos_y = vio_wob_pos_y + gt_pos_y(1) - vio_wob_pos_y(1);
vio_wob_pos_z = vio_wob_pos_z + gt_pos_z(1) - vio_wob_pos_z(1);
% rotate data
angle = -1.9/180*pi;
R = [cos(angle)   -sin(angle)  0;
    sin(angle)    cos(angle)  0;
    0                   0     1];
rotated = R * [vio_wob_pos_x';vio_wob_pos_y';vio_wob_pos_z'];
vio_wob_pos_x = rotated(1,:)';
vio_wob_pos_y = rotated(2,:)';
vio_wob_pos_z = rotated(3,:)';


%calculate vilo wob velocity
vio_wob_vel_x = (vio_wob_pos_x(2:end) - vio_wob_pos_x(1:end-1))./(ref_time(2:end)-ref_time(1:end-1));
vio_wob_vel_y = (vio_wob_pos_y(2:end) - vio_wob_pos_y(1:end-1))./(ref_time(2:end)-ref_time(1:end-1));
vio_wob_vel_z = (vio_wob_pos_z(2:end) - vio_wob_pos_z(1:end-1))./(ref_time(2:end)-ref_time(1:end-1));
% smooth the velocity
vio_wob_vel_x = movmean(vio_wob_vel_x,2,1);
vio_wob_vel_y = movmean(vio_wob_vel_y,2,1);
vio_wob_vel_z = movmean(vio_wob_vel_z,2,1);

%% read vilo wb pos
% interpolate time to get all pos
vio_wb_pos_x = interp1(vio_wb_time,vio_wb_Tab.Var2,ref_time);
vio_wb_pos_y = interp1(vio_wb_time,vio_wb_Tab.Var3,ref_time);
vio_wb_pos_z = interp1(vio_wb_time,vio_wb_Tab.Var4,ref_time);
% move data to align with ground truth
vio_wb_pos_x = vio_wb_pos_x + gt_pos_x(1) - vio_wb_pos_x(1);
vio_wb_pos_y = vio_wb_pos_y + gt_pos_y(1) - vio_wb_pos_y(1);
vio_wb_pos_z = vio_wb_pos_z + gt_pos_z(1) - vio_wb_pos_z(1);
% rotate data
angle = -1.9/180*pi;
R = [cos(angle)   -sin(angle)  0;
    sin(angle)    cos(angle)  0;
    0                   0     1];
rotated = R * [vio_wb_pos_x';vio_wb_pos_y';vio_wb_pos_z'];
vio_wb_pos_x = rotated(1,:)';
vio_wb_pos_y = rotated(2,:)';
vio_wb_pos_z = rotated(3,:)';

angle = 3.9/180*pi;
R = [cos(angle)  0  -sin(angle) ;
    0  1  0  ;
    sin(angle)           0        cos(angle)];
rotated = R * [vio_wb_pos_x';vio_wb_pos_y';vio_wb_pos_z'];
vio_wb_pos_x = rotated(1,:)';
vio_wb_pos_y = rotated(2,:)';
vio_wb_pos_z = rotated(3,:)';

%calculate vilo wob velocity
vio_wb_vel_x = (vio_wb_pos_x(2:end) - vio_wb_pos_x(1:end-1))./(ref_time(2:end)-ref_time(1:end-1));
vio_wb_vel_y = (vio_wb_pos_y(2:end) - vio_wb_pos_y(1:end-1))./(ref_time(2:end)-ref_time(1:end-1));
vio_wb_vel_z = (vio_wb_pos_z(2:end) - vio_wb_pos_z(1:end-1))./(ref_time(2:end)-ref_time(1:end-1));
% smooth the velocity
vio_wb_vel_x = movmean(vio_wb_vel_x,2,1);
vio_wb_vel_y = movmean(vio_wb_vel_y,2,1);
vio_wb_vel_z = movmean(vio_wb_vel_z,2,1);


%% plot all z pos

fontsize = 18
figure(1);clf;hold on;
set(gca,'FontSize',fontsize)
plot(ref_time, gt_pos_z,'LineWidth',2);
plot(ref_time, vio_pos_z,'LineWidth',2);
plot(ref_time, vio_wob_pos_z,'LineWidth',2);
plot(ref_time, vio_wb_pos_z,'LineWidth',2);
legend('Ground Truth', 'VIO', "VILO+NoBiasCorrect", "VILO+BiasCorrect", 'Location','southeast')
xlabel('Time (s)')
ylabel('Z Position (m)')
hold off;
set(gcf, 'Position', [214 1703 550 420])

%% plot all trajectory 
figure(2);clf;
set(gca,'FontSize',fontsize)
plot3(gt_pos_x, gt_pos_y, gt_pos_z,'LineWidth',2);hold on;
plot3(vio_pos_x, vio_pos_y, vio_pos_z,'LineWidth',2);hold on;
plot3(vio_wob_pos_x, vio_wob_pos_y, vio_wob_pos_z,'LineWidth',2);hold on;
plot3(vio_wb_pos_x, vio_wb_pos_y, vio_wb_pos_z,'LineWidth',2);hold off;
axis equal
legend('Ground Truth', 'VIO', "VILO+NoBiasCorrect", "VILO+BiasCorrect", 'Location','southeast')
set(gcf, 'Position', [12 1150 1581 464])


%% plot all x velocity
figure(3);clf;
subplot(3,1,1)
plot(ref_time(1:end-1), gt_vel_x,'LineWidth',2);hold on;
plot(ref_time(1:end-1), lo_vel_x(1:end-1),'LineWidth',2);
plot(ref_time(1:end-1), vio_vel_x,'LineWidth',2);
plot(ref_time(1:end-1), vio_wob_vel_x,'LineWidth',2);
plot(ref_time(1:end-1), vio_wb_vel_x,'LineWidth',2);
ax = gca;ax.XLim = [20 35];ax.YLim = [-0.2 0.5];
set(gca,'FontSize',fontsize)
xlabel('Time (s)')
ylabel('X Velocity (m/s)')
legend('Ground Truth', 'LO', 'VIO',  "VILO+NoBiasCorrect", "VILO+BiasCorrect", 'Location','northwest', 'FontSize', 14)
subplot(3,1,2)
plot(ref_time(1:end-1), gt_vel_y,'LineWidth',2);hold on;
plot(ref_time(1:end-1), lo_vel_y(1:end-1),'LineWidth',2);
plot(ref_time(1:end-1), vio_vel_y,'LineWidth',2);
plot(ref_time(1:end-1), vio_wob_vel_y,'LineWidth',2);
plot(ref_time(1:end-1), vio_wb_vel_y,'LineWidth',2);
set(gca,'FontSize',fontsize)
ax = gca;ax.XLim = [20 35];ax.YLim = [-0.5 0.5];
xlabel('Time (s)')
ylabel('Y Velocity (m/s)')
legend('Ground Truth', 'LO', 'VIO',  "VILO+NoBiasCorrect", "VILO+BiasCorrect", 'Location','northwest', 'FontSize', 14)
subplot(3,1,3)
plot(ref_time(1:end-1), gt_vel_z,'LineWidth',2);hold on;
plot(ref_time(1:end-1), lo_vel_z(1:end-1),'LineWidth',2);
plot(ref_time(1:end-1), vio_vel_z,'LineWidth',2);
plot(ref_time(1:end-1), vio_wob_vel_z,'LineWidth',2);
plot(ref_time(1:end-1), vio_wb_vel_z,'LineWidth',2);
set(gca,'FontSize',fontsize)
ax = gca;ax.XLim = [20 35]; ax.YLim = [-0.3 0.3];
xlabel('Time (s)') 
ylabel('Z Velocity (m/s)')
legend('Ground Truth', 'LO', 'VIO',  "VILO+NoBiasCorrect", "VILO+BiasCorrect", 'Location','northwest', 'FontSize', 14)
set(gcf, 'Position', [20 27 1567 1038])

%%
figure(4);clf;
subplot(3,1,1)
plot(ref_time(1:end-1), gt_vel_x - vio_vel_x,'LineWidth',2);hold on;
plot(ref_time(1:end-1), gt_vel_x - vio_wob_vel_x,'LineWidth',2);
plot(ref_time(1:end-1), gt_vel_x - vio_wb_vel_x,'LineWidth',2);
xlabel('Time')
title('X Velocity Difference')
legend('VIO', "VILO+NoBiasCorrect", "VILO+BiasCorrect", 'Location','northwest')
subplot(3,1,2)
plot(ref_time(1:end-1), gt_vel_y - vio_vel_y,'LineWidth',2);hold on;
plot(ref_time(1:end-1), gt_vel_y - vio_wob_vel_y,'LineWidth',2);
plot(ref_time(1:end-1), gt_vel_y - vio_wb_vel_y,'LineWidth',2);
xlabel('Time')
title('Y Velocity Difference')
legend('VIO', "VILO+NoBiasCorrect", "VILO+BiasCorrect", 'Location','northwest')
subplot(3,1,3)
plot(ref_time(1:end-1), gt_vel_z - vio_vel_z,'LineWidth',2);hold on;
plot(ref_time(1:end-1), gt_vel_z - vio_wob_vel_z,'LineWidth',2);
plot(ref_time(1:end-1), gt_vel_z - vio_wb_vel_z,'LineWidth',2);
xlabel('Time')
title('Z Velocity Difference')
legend('VIO', "VILO+NoBiasCorrect", "VILO+BiasCorrect", 'Location','northwest')
set(gcf, 'Position', [20 27 1567 1038])


display('------x------')
k = abs(gt_vel_x - lo_vel_x(1:end-1));
k(isnan(k))=0;
% max(k)
sqrt(sum(k'*k)/size(k,1))
k = abs(gt_vel_x - vio_vel_x);
% max(k)
sqrt(sum(k'*k)/size(k,1))

k = abs(gt_vel_x - vio_wob_vel_x);
% max(k)
sqrt(sum(k'*k)/size(k,1))
k = abs(gt_vel_x - vio_wb_vel_x);
% max(k)
sqrt(sum(k'*k)/size(k,1))

display('------y------')
k = abs(gt_vel_y - lo_vel_y(1:end-1));
k(isnan(k))=0;
% max(k)
sqrt(sum(k'*k)/size(k,1))
k = abs(gt_vel_y - vio_vel_y);
% max(k)
sqrt(sum(k'*k)/size(k,1))
k = abs(gt_vel_y - vio_wob_vel_y);
% max(k)
sqrt(sum(k'*k)/size(k,1))
k = abs(gt_vel_y - vio_wb_vel_y);
% max(k)
sqrt(sum(k'*k)/size(k,1))


display('------z------')
k = abs(gt_vel_z - lo_vel_z(1:end-1));
k(isnan(k))=0;
% max(k)
sqrt(sum(k'*k)/size(k,1))
k = abs(gt_vel_z - vio_vel_z);
% max(k)
sqrt(sum(k'*k)/size(k,1))
k = abs(gt_vel_z - vio_wob_vel_z);
% max(k)
sqrt(sum(k'*k)/size(k,1))
k = abs(gt_vel_z - vio_wb_vel_z);
% max(k)
sqrt(sum(k'*k)/size(k,1))
