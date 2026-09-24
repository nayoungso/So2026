function RT_behav_ME0_singleTrace(date,taskid)
% RT_behav_ME0_singleTrace
%
% Plots the session-averaged motion-energy (ME) trace, aligned to dots-on
% and to saccade onset, pooled across all trials at zero coherence for a
% given monkey/taskid.
%
% ME is re-signed per trial so that positive ME always reflects motion
% energy supporting the monkey's actual choice (mME as stored is signed
% relative to t1_dir; trials choosing T2 get flipped), then both choice
% conditions are combined into a single trace and averaged across
% trials (mean +/- SEM).
%
% REQUIRES ON PATH: shadedErrorBar.m and fig_setting.m (included
% alongside this file).
%
% INPUTS
%   date   : monkey/session identifier string, used as a dir() wildcard
%            prefix (dir([path date '*_behav_me.mat'])) -- pass a
%            monkey letter (e.g. 'H', 'N') to pool across all of that
%            monkey's session files, or a full session string to
%            restrict to one session.
%   taskid : trials are included if trials(j).taskid <= taskid (a
%            cutoff, not exact membership) AND trials(j).response>=0 AND
%            trials(j).dot_coh==0 AND the trial has a non-empty cME
%            field.
%
% OUTPUT
%   None (produces figure 4: left panel = ME vs. time from dots-on;
%   right panel = ME vs. time from saccade).
%
% EXAMPLE
%   RT_behav_ME0_singleTrace('N', 21)

clear trials_temp

path = ['~/So2026/BehaviorOnly/'];

filename = dir(strcat(path,date,'*_behav_me.mat'))

trialno = 0;


for i = 1:length(filename)

    clear trials
    filename(i).name
    load([path,filename(i).name]);

    for j = 1:length(trials)


        %if (trials(j).taskid == taskid)&(trials(j).response>=0)&(trials(j).dot_coh == 0)&~isempty(trials(j).cME)
        if (trials(j).taskid <= taskid)&(trials(j).response>=0)&(trials(j).dot_coh == 0)&~isempty(trials(j).cME)

                trialno = trialno+1;
                trials_temp(trialno) = trials(j);
                sessionid(trialno) = i;
        end

    end



end

clear trials
trials = trials_temp;
clear trials_temp
length(trials)


% ME is re-signed per trial so that positive ME always reflects motion
% energy supporting the monkey's actual choice (mME as stored is signed
% relative to t1_dir; trials choosing t2 get flipped), then both choice
% conditions are combined into a single trace.
trialno_me = 0;
ME = nan(length(trials),600);
ME_sacc = nan(length(trials),600);

RT = nan(1,length(trials));
choice_sessionid = nan(1,length(trials));

RT_session = cell(length(filename),1);

for i = 1:length(trials)

    %dir_sign = floor(mod(trials(i).dot_dir,360)/180);  % 0 (e.g., 0 deg) or 1 (e.g., 180 deg)
    dir_sign = trials(i).dot_dir == trials(i).t1_dir;
    rt = trials(i).time_sacc(1) - trials(i).time_dots_on(1);
    rt_frno = round(rt/(1000/75));

    try
        if ((trials(i).response == 1)&&(dir_sign == 1))||((trials(i).response == 0)&&(dir_sign == 0))       %t1 choice
            me_sign = 1;
        else                                                                                                  %t2 choice
            me_sign = -1;
        end

        trialno_me = trialno_me+1;
        ME(trialno_me,1:length(trials(i).mME{1})) = me_sign * trials(i).mME{1};

        ME_sacc(trialno_me,200-rt_frno:200-rt_frno+length(trials(i).mME{1})-1) = me_sign * trials(i).mME{1};

        RT(trialno_me) = trials(i).time_sacc(1) - trials(i).time_dots_on(1);
        choice_sessionid(trialno_me) = sessionid(i);

        if isempty(RT_session{sessionid(i)})
            trialno_session = 1;
        else
            trialno_session = trialno_session+1;
        end
        RT_session{sessionid(i)}(trialno_session) = trials(i).time_sacc(1) - trials(i).time_dots_on(1);

    catch
        rt_frno
    end

end

ME_avg = nanmean(ME,1);
ME_sacc_avg = nanmean(ME_sacc,1);

%size(isnan(ME))
ME_se = nanstd(ME,1)./sqrt(sum(~isnan(ME),1));
ME_sacc_se = nanstd(ME_sacc,1)./sqrt(sum(~isnan(ME_sacc),1));

clear early_id late_id
for i = 1:length(filename)

    RT_bnd = quantile(RT_session{i},[1/3 2/3]);

    if i == 1
        early_id = (choice_sessionid == i)&(RT<=RT_bnd(1));
        sum(early_id)
    else
        early_id = early_id + ((choice_sessionid == i)&(RT<=RT_bnd(1)));
    end

    if exist('late_id')
        late_id = late_id + ((choice_sessionid == i)&(RT>=RT_bnd(2)));
    else
        late_id = (choice_sessionid == i)&(RT>=RT_bnd(2));
    end
end


early = logical(early_id);
late = logical(late_id);

sum(early)
sum(late)

early_ME_avg = nanmean(ME(early,:),1);
late_ME_avg = nanmean(ME(late,:),1);

early_ME_se = nanstd(ME(early,:),1)./sqrt(sum(~isnan(ME(early,:)),1));
late_ME_se = nanstd(ME(late,:),1)./sqrt(sum(~isnan(ME(late,:)),1));


early_ME_sacc_avg = nanmean(ME_sacc(early,:),1);
late_ME_sacc_avg = nanmean(ME_sacc(late,:),1);

early_ME_sacc_se = nanstd(ME_sacc(early,:),1)./sqrt(sum(~isnan(ME_sacc(early,:)),1));
late_ME_sacc_se = nanstd(ME_sacc(late,:),1)./sqrt(sum(~isnan(ME_sacc(late,:)),1));



timex = 1000/75:1000/75:length(ME_avg)*1000/75;

%timex_sacc = (200-80)*(1000/75):1000/75:(200+30)*(1000/75);
timex_sacc = -80*(1000/75):1000/75:30*(1000/75);
% to plot ME_sacc_avg(200-80:200+30)

figure(4)
subplot(1,2,1)
shadedErrorBar(timex,ME_avg,ME_se,'k-');

xlim([0 1500]);
xlabel('time from dots on (msec)');
%grid on
fig_setting();
ylabel('Motion Energy (a.u.)');

subplot(1,2,2)
shadedErrorBar(timex_sacc,ME_sacc_avg(200-80:200+30),ME_sacc_se(200-80:200+30),'k-');
line([0 0],[-10 10],'Color','k')
xlim([-1200 200]);
xlabel('time from sacc (msec)');
ylabel('Motion Energy (a.u.)');
%suptitle(foldername)
%grid on
fig_setting();

%{
figure(5)
subplot(121)
shadedErrorBar(timex,early_ME_avg, early_ME_se,'b-');
hold on
shadedErrorBar(timex,late_ME_avg,late_ME_se,'r-');

hold off
xlim([0 1500]);
xlabel('time from dots on (msec)');
grid on
ylabel('Motion Energy (a.u.)');



subplot(122)
shadedErrorBar(timex_sacc,early_ME_sacc_avg(200-80:200+30),early_ME_sacc_se(200-80:200+30),'b-');
hold on
shadedErrorBar(timex_sacc,late_ME_sacc_avg(200-80:200+30),late_ME_sacc_se(200-80:200+30),'r-');
line([0 0],[-10 10],'Color','k')
hold off
xlim([-800 200]);
xlabel('time from sacc (msec)');
ylabel('Motion Energy (a.u.)');
%suptitle(foldername)
grid on
%}

end