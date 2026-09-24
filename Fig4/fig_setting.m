function fig_setting(ymax, offset_ymax)

if nargin == 2
    ylim([0, ymax + offset_ymax]);
end

box off;
set(gca, 'TickDir', 'out');
