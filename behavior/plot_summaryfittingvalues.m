% Linear fitting values
% naive control: 0.036, R2 = 0.675
% lesion contra side pre:  0.038 R2 1.043
% lesion contra side post: 0.034 R2 0.7885
% lesion lesion side pre:  0.024 R2 0.8275
% lesion lesion side post: 0.054 R2 0.9263
% optopre stimulated control    0.028 
% optopre stimulated stimulated 0.062
% optopre contra control        0.028
% optopre contra stimulated     0.066
% optopost stimulated control    0.045
% optopost stimulated stimulated 0.065
% optopost contra control        0.045
% optopost contra stimulated     0.038

values = [0.036 0.038 0.034 0.024 0.054 0.028 0.062 0.028 0.066 0.045 0.065 0.045 0.038 ];
valuesdiff = [0.038-0.034 0.024-0.054 0.028-0.062 0.028-0.066 0.045-0.065 0.045-0.038 ];
figure
subplot(2,2,1)
bar(abs(valuesdiff))

