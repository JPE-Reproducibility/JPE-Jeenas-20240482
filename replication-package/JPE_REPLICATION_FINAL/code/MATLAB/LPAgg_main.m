% Matlab code used for aggregate local projection estimation in the paper
% "Firm Balance Sheet Liquidity, Monetary Policy Shocks, and Investment Dynamics" (by Priit Jeenas)
% 
% These codes, methods, and the data sources on U.S. aggregate output, consumption, investment and Fed funds rate are closely derived from the replication package accompanying the paper: Luetticke, R. (2021). Transmission of monetary policy with heterogeneity in household portfolios. American Economic Journal: Macroeconomics, 13(2), 1-25.

% Define the root directory, just in case, conditional on execution of file
this_file = mfilename('fullpath');
PROJECT_ROOT = fileparts(fileparts(fileparts(this_file)));
cd(PROJECT_ROOT);
addpath(genpath(fullfile(PROJECT_ROOT, 'code', 'MATLAB')));

%% Speficy lags and start data
L=21; %lag length 

startyear = 1990;
startdate1 = (startyear-1968.75)*4+1; % Correct start entry for aggregate series

%% Load aggregate data (as in Luetticke, 2021)
% All data is 1966Q1-2013Q1
Consumption_raw = xlsread(fullfile(PROJECT_ROOT, 'data', 'raw', 'LPAgg_data', 'PCECC96.xlsx'),'Quarterly','','basic');
Consumption = Consumption_raw(88:276,2); % Select the correct data rows
Investment_raw = xlsread(fullfile(PROJECT_ROOT, 'data', 'raw', 'LPAgg_data', 'GPDIC1.xlsx'),'Quarterly','','basic');
Investment = Investment_raw(88:276,2); % Select the correct data rows
GovSpending_raw = xlsread(fullfile(PROJECT_ROOT, 'data', 'raw', 'LPAgg_data', 'GCEC1.xlsx'),'Quarterly','','basic');
GovSpending = GovSpending_raw(88:276,2); % Select the correct data rows
% As Luetticke (2021), consider output as Y=C+I+G
Output = Consumption + Investment + GovSpending;

% Load consumption deflator
CDeflev_xls = xlsread(fullfile(PROJECT_ROOT, 'data', 'raw', 'LPAgg_data', 'DNDGRD3Q086SBEA.xlsx'),'Quarterly','','basic'); % Personal consumption expenditures: Nondurable goods
CDeflev = CDeflev_xls(88:276,2); % CDeflator level

% Load investment deflator(s)
CIDeflator_xls = xlsread(fullfile(PROJECT_ROOT, 'data', 'raw', 'LPAgg_data', 'A006RD3Q086SBEA.xlsx'),'Quarterly','','basic'); % Gross private domestic investment 
STRIDeflator_xls = xlsread(fullfile(PROJECT_ROOT, 'data', 'raw', 'LPAgg_data', 'A009RD3Q086SBEA.xlsx'),'Quarterly','','basic'); % Gross private domestic investment: Fixed investment: Nonresidential: Structures
EQIDeflator_xls = xlsread(fullfile(PROJECT_ROOT, 'data', 'raw', 'LPAgg_data', 'Y033RD3Q086SBEA.xlsx'),'Quarterly','','basic'); % Gross private domestic investment: Fixed investment: Nonresidential: Equipment

% ApplyDef = Deflator;
ApplyDef = (CDeflev_xls(88:276,2)./CDeflev_xls(87:275,2)); % Nondurable deflator growth
CIDeflator = (CIDeflator_xls(88:276,2)./CIDeflator_xls(87:275,2)) ./ ApplyDef;          % CIdeflator gross growth, relative to CDeflator
STRIDeflator = (STRIDeflator_xls(88:276,2)./STRIDeflator_xls(87:275,2)) ./ ApplyDef;    % STRIdeflator gross growth, relative to CDeflator
EQIDeflator = (EQIDeflator_xls(88:276,2)./EQIDeflator_xls(87:275,2)) ./ ApplyDef;       % EQIdeflator gross growth, relative to CDeflator

%% Load policy rate and monetary shocks
% Shocks to nominal interest rate 1969-2007
N=2007-1969+1;
% Federal funds rate
FFR=xlsread(fullfile(PROJECT_ROOT, 'data', 'raw', 'FEDFUNDS_Q.xls'),'FRED Graph','','basic');
FedFunds=FFR(58:246,2);

% Load Jarocinski and Karadi (2024) FF4 shocks, as used throughout
len_shock_series_sample = (2008-startyear)*4;
len_shock_series_full   = (2017-startyear)*4;    
JKTable = readtable(fullfile(PROJECT_ROOT, 'data', 'raw', 'JK_data_fig4.csv'), 'Delimiter', ',', 'MissingRule', 'fill');
JK_shock_series = zeros(len_shock_series_full,1);
JK_shock_series(1,1) = sum(table2array(JKTable(1:2,3)));
for tt=2:length(JK_shock_series)
    JK_shock_series(tt) = sum(table2array(JKTable( (3 + 3*(tt-2)):(3 + 3*(tt-2)+2) ,3)));
end
JK_STDEVSHOCK= std(JK_shock_series);
shock_series_normed = JK_shock_series(1:len_shock_series_sample)/JK_STDEVSHOCK;


%% Local projection: Including a linear time trend
Ymat1=100*[log(Output) log(Consumption) log(Investment) FedFunds/100 ];
Ymat=Ymat1(startdate1:end,:);

% Select dependent variables    
Ymat=[Ymat 100*log(STRIDeflator(startdate1:end,:)) 100*log(EQIDeflator(startdate1:end,:)) 100*log(CIDeflator(startdate1:end,:)) ];

% Select controls
Zmat=[Ymat1((startdate1-1):(startdate1-2+length(shock_series_normed)),1:4 )]; %Lagged Y as controls
X=[shock_series_normed ones(max(size(shock_series_normed)),1) (1:max(size(shock_series_normed)))' Zmat ];

% Run local projections
[T,nvar]=size(Ymat((1:size(X,1)),:));
[~,nvarX]=size(X);
FY=nan(T,nvar,L);
Data=nan(T,nvar+nvarX,L);
IRF=nan(size(Ymat,2),L);
    
for j=1:nvar
    for i=0:1:L-1
               
        FY(:,:,i+1)=[Ymat(1+i:length(shock_series_normed)+i,:)];        
        Data(:,:,i+1)=[FY(:,:,i+1) X];
        [beta,bint,r,rint,stats] =regress(FY(:,j,i+1),X);
        IRF(j,i+1)=beta(1);              
    end
end
% Cumulative sum of I inflation path
IRF_csSTRIdef = cumsum(IRF(5,:));
IRF_csEQIdef = cumsum(IRF(6,:));
IRF_csCIdef = cumsum(IRF(7,:));

%% Bootstrap confidence intervals
% Set seed
rng(1)
%
[n,numv]= size(Data);
bootstraps=1000;
indexes = randi(n,n,bootstraps);

BIRF=nan(size(Ymat,2),L,bootstraps);
% Also, bootstrap cumulative sum of I inflation path
BIRF_csSTRIdef = nan(L,bootstraps);
BIRF_csEQIdef = nan(L,bootstraps);
BIRF_csCIdef = nan(L,bootstraps);
for bb=1:bootstraps
    for j=1:size(Ymat,2)
        for i=0:1:L-1
        BData = Data(indexes(:,bb),:,i+1);
        [beta,bint,r,rint,stats] =regress(BData(:,j),BData(:,nvar+1:end));
        BIRF(j,i+1,bb)=beta(1);
        end
    end
    BIRF_csSTRIdef(:,bb) = cumsum(BIRF(5,:,bb));
    BIRF_csEQIdef(:,bb) = cumsum(BIRF(6,:,bb));
    BIRF_csCIdef(:,bb) = cumsum(BIRF(7,:,bb));    
end
Ll_conf = 5;
Ul_conf = 95;
IRF_conf(1,:,:)=1*prctile(BIRF,Ll_conf,3);
IRF_conf(2,:,:)=1*prctile(BIRF,Ul_conf,3);
IRFscaled=1*IRF;
csSTRIdef_conf(1,:)=1*prctile(BIRF_csSTRIdef,Ll_conf,2);
csSTRIdef_conf(2,:)=1*prctile(BIRF_csSTRIdef,Ul_conf,2);
csEQIdef_conf(1,:)=1*prctile(BIRF_csEQIdef,Ll_conf,2);
csEQIdef_conf(2,:)=1*prctile(BIRF_csEQIdef,Ul_conf,2);
csCIdef_conf(1,:)=1*prctile(BIRF_csCIdef,Ll_conf,2);
csCIdef_conf(2,:)=1*prctile(BIRF_csCIdef,Ul_conf,2);

%% Plotting Figures
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');
set(groot, 'defaultTextInterpreter','latex');
set(groot, 'defaultFigureVisible', 'off');

% Create folder for output
outdir = fullfile(PROJECT_ROOT, 'output', 'figures', 'empirics_LPAgg');
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

% Also, load model IRFs
IRF_mod = readtable(fullfile(PROJECT_ROOT, 'interim_output', 'model_interim_output', 'IRFs_base.csv'), 'Delimiter', ',', 'MissingRule', 'fill');
IRF_mod_rb = table2array(IRF_mod(:,1));
IRF_mod_Y = table2array(IRF_mod(:,2));
IRF_mod_I = table2array(IRF_mod(:,3));
IRF_mod_Q = table2array(IRF_mod(:,4));
mod_scaler = IRFscaled(4,1)/IRF_mod_rb(2); % Scale model IRFs, with rf proportionally to fed funds rate impact response

controls='LaggedYmat';

figsizevar = [0.175 0.15 0.60 0.50];

figurename=['FigA15b_IRF_YI_LPAgg'];
figure('Name',figurename,'NumberTitle','off','Position',[1 1 720 720]);
l1=plot([0:1:(L-1)],IRFscaled(3,:),'b-','LineWidth',4.5, 'DisplayName', 'Investment');
hold on;
plot([0:1:(L-1)],squeeze(IRF_conf(1,3,:)),'b--','LineWidth',1.5);
plot([0:1:(L-1)],squeeze(IRF_conf(2,3,:)),'b--','LineWidth',1.5);
l2=plot([0:1:(L-1)],IRFscaled(1,:),'-.','LineWidth',4.5, 'Color', [0.48 0.68 0.19], 'DisplayName', 'Output');
plot([0:1:(L-1)],squeeze(IRF_conf(1,1,:)),'--','LineWidth',1.5, 'Color', [0.48 0.68 0.19]);
plot([0:1:(L-1)],squeeze(IRF_conf(2,1,:)),'--','LineWidth',1.5, 'Color', [0.48 0.68 0.19]);
xlim([0 L-1]);
ylim([-5.7 1.3]);
xticks([0 4 8 12 16 20]);
ylabel('Percent','FontSize',25);
xlabel('Quarter ($h$)','FontSize',25);
hold on;
plot(0:L,zeros(1,L+1),'--black');
set(gca, 'FontName', 'arial','FontSize',25);
set(gca,'Position', figsizevar);
legend([l1 l2], 'Location', 'southwest','FontSize',25);
ytickformat('%.1f');
grid on;
% printpdf(gcf,['figs_out/' figurename])
l4=plot([0:1:(L-1)],mod_scaler*IRF_mod_Y,'k-.','LineWidth',3, 'DisplayName', 'Output (Model)');
l5=plot([0:1:(L-1)],mod_scaler*IRF_mod_I,'k:','LineWidth',3, 'DisplayName', 'Investment (Model)');
legend([l1 l2 l5 l4], 'Location', 'southwest','FontSize',25, 'NumColumns',2);
outfile = fullfile(outdir, [figurename '_wmod']);
printpdf(gcf, outfile);

figurename=['FigA15a_IRF_FedFunds_LPAgg'];
figure('Name',figurename,'NumberTitle','off','Position',[1 1 720 720]);
l1=plot([0:1:(L-1)],IRFscaled(4,:),'b-','LineWidth',4.5);
hold on;
plot([0:1:(L-1)],squeeze(IRF_conf(1,4,:)),'b--','LineWidth',2);
plot([0:1:(L-1)],squeeze(IRF_conf(2,4,:)),'b--','LineWidth',2);
xlim([0 L-1]);
xticks([0 4 8 12 16 20]);
ylabel('Percentage points','FontSize',25);
xlabel('Quarter ($h$)','FontSize',25);
hold on;
plot(0:L,zeros(1,L+1),'--black');
set(gca, 'FontName', 'arial','FontSize',25);
set(gca,'Position', figsizevar);
ytickformat('%.1f');
grid on;
% printpdf(gcf,['figs_out/' figurename])
l2=plot([0:1:(L-2)],mod_scaler*IRF_mod_rb(2:end),'k:','LineWidth',3);
legend([l1 l2], {'Fed funds rate', '$r_{t+1}^{f}$ (Model)'}, 'Location', 'southwest','FontSize',25, 'NumColumns',1, 'Interpreter', 'latex');
outfile = fullfile(outdir, [figurename '_wmod']);
printpdf(gcf, outfile);

figurename=['FigA15c_IRF_CIDeflatorCS_LPAgg'];
figure('Name',figurename,'NumberTitle','off','Position',[1 1 720 720]);
l1=plot([0:1:(L-1)],IRF_csCIdef,'b-','LineWidth',4.5);
hold on;
plot([0:1:(L-1)],squeeze(csCIdef_conf(1,:)),'b--','LineWidth',2);
plot([0:1:(L-1)],squeeze(csCIdef_conf(2,:)),'b--','LineWidth',2);
xlim([0 L-1]);
xticks([0 4 8 12 16 20]);
ylabel('Percent','FontSize',25);
xlabel('Quarter ($h$)','FontSize',25);
hold on;
plot(0:L,zeros(1,L+1),'--black');
set(gca, 'FontName', 'arial','FontSize',25);
set(gca,'Position', figsizevar);
ytickformat('%.1f');
grid on;
% printpdf(gcf,['figs_out/' figurename])
l2=plot([0:1:(L-1)],mod_scaler*IRF_mod_Q,'k:','LineWidth',3);
legend([l1 l2], {'$P_{t}^{I}$', '$Q_{t}$ (Model)'}, 'Location', 'southwest','FontSize',25, 'NumColumns',1, 'Interpreter', 'latex');
outfile = fullfile(outdir, [figurename '_wmod']);
printpdf(gcf, outfile);

figurename=['FigA15d_IRF_ALLIDeflatorCS_LPAgg'];
figure('Name',figurename,'NumberTitle','off','Position',[1 1 720 720]);
l1=plot([0:1:(L-1)],IRF_csSTRIdef,'b-','LineWidth',4.5);
hold on;
plot([0:1:(L-1)],squeeze(csSTRIdef_conf(1,:)),'b--','LineWidth',1.5);
plot([0:1:(L-1)],squeeze(csSTRIdef_conf(2,:)),'b--','LineWidth',1.5);
l2=plot([0:1:(L-1)],IRF_csEQIdef,'-.','LineWidth',4.5, 'Color', [0.64 0.08 0.18]);
plot([0:1:(L-1)],squeeze(csEQIdef_conf(1,:)),'--','LineWidth',1.5, 'Color', [0.64 0.08 0.18]);
plot([0:1:(L-1)],squeeze(csEQIdef_conf(2,:)),'--','LineWidth',1.5, 'Color', [0.64 0.08 0.18]);
xlim([0 L-1]);
xticks([0 4 8 12 16 20]);
ylabel('Percent','FontSize',25);
xlabel('Quarter ($h$)','FontSize',25);
hold on;
plot(0:L,zeros(1,L+1),'--black');
set(gca, 'FontName', 'arial','FontSize',25);
set(gca,'Position', figsizevar);
legend([l1 l2], {'$P_{t}^{I}$ (Structures)', '$P_{t}^{I}$ (Equipment)'}, 'Location', 'southwest','FontSize',25, 'NumColumns',1, 'Interpreter', 'latex');
ytickformat('%.1f');
grid on;
outfile = fullfile(outdir, [figurename '_wmod']);
printpdf(gcf, outfile);


