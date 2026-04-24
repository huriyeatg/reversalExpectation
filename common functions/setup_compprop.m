% Add your computer setting

switch upper(char(java.net.InetAddress.getLocalHost.getHostName))
    case 'KWANLAB-HA'
        disp('Huriye''s PC in George St.');
        root_path = 'E:\MATLAB\two-lickport-projects\';
    case 'WIN-AMP016'
        disp('Huriye''s PC in Sherrington Road.');
        root_path = 'C:\Users\Huriye\Documents\code\bandit';
    case 'MWMJ046RNP'
        disp('C.Murphy''s PC in George St.');
        root_path = 'E:\MATLAB\PRBehaviour\bandit-master\';
    case 'MWMJ0A8Y18'
        disp('Heathers lab comp');
        root_path = 'C:\Users\ho83\Documents\MATLAB\bandit2020\bandit';    
    case 'HURIYEMAC.LOCAL'
        disp('Huriye New MAC Computer');
        root_path = '/Users/Huriye/Documents/Code/bandit';  
    case 'DESKTOP-COVVGR0'
        disp('Seyma''s Laptop');
        root_path = 'F:\Bandit Longitudinal Analysis\Bandit Github\bandit';
    case 'HAKKI_PC'
        disp('Seyma''s PC in Umram');
        root_path = 'E:\Bandit Longitudinal Analysis\Bandit Github\bandit';
    otherwise
        disp('Some unknown computer');
        root_path = input (' Enter your path for your bandit-master code:');
end