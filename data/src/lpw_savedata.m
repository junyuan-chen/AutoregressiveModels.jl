% Modified from Li, Plagborg-Møller and Wolf (2024) for testing purposes
% Estimation part is based on Stock and Watson (2016)
% To run the code, download the replication repo from
% https://github.com/dake-li/lp_var_simul
% Place this file in the DFM folder
% Data for testing purposes are saved to lpw_est_data.mat


%% Content from run_dfm.m

%% HOUSEKEEPING

clc
clear all
close all

addpath(genpath(fullfile('..', 'Auxiliary_Functions')))
addpath(genpath(fullfile('..', 'Estimation_Routines')))
addpath(genpath('Subroutines'))

%% SET EXPERIMENT

lag_type = 4; % No. of lags to impose in estimation, or NaN (= AIC)

%% SETTINGS

% Apply shared settings as well as settings specific to DGP and estimand type

run(fullfile('Settings', 'shared'));

%% ENCOMPASSING DFM MODEL

%% Set the argument values for DFM_est

n_factors = DF_model.n_fac;
n_lags_fac = DF_model.n_lags_fac;
n_lags_uar = DF_model.n_lags_uar;
reorder = DF_model.reorder;
levels = 1;
coint_rank = DF_model.coint_rank;


%% Content from DFM_est.m

%% PREPARATIONS

small = 1.0e-10;
big = 1.0e+6;

% ----------- Sample Period, Calendars and so forth
[dnobs_m,calvec_m,calds_m] = calendar_make([1959 1],[2014 12],12);  % Monthly Calendar
[dnobs_q,calvec_q,calds_q] = calendar_make([1959 1],[2014 4],4);    % Quarterly Calendar

% -- Load Data
load_data=1;
datain_all;      % datain_all reads in the full dataset .. all variables, etc. saved in datain.xx

%% ESTIMATION

% Factor Parameters
n_fac = n_factors;
est_par.fac_par.nfac.unobserved = n_fac;
est_par.fac_par.nfac.observed = 0;
est_par.fac_par.nfac.total = n_fac + est_par.fac_par.nfac.observed;

% Sampling parameters
est_par.smpl_par.nfirst = [1959 3];       % start date
est_par.smpl_par.nlast  = [2014 4];       % end date
est_par.smpl_par.calvec = datain.calvec;  % calendar
est_par.smpl_par.nper   = 4;              % number of periods a year

% Factor analysis parameters
est_par.fac_par.nt_min                  = 20;     % min number of obs for any series used to est factors
est_par.lambda.nt_min                   = 40;     % min number of obs for any series used to estimate lamba, irfs, etc.
est_par.fac_par.tol                     = 10^-8;  % precision of factor estimation (scaled by by n*t)

% Restrictions on factor loadings to identify factors
est_par.fac_par.lambda_constraints_est  = 1;  % no constraints on lambda
est_par.fac_par.lambda_constraints_full = 1;  % no constraints on lambda

% VAR parameters for factors
est_par.var_par.nlag   = n_lags_fac;    % number of lags
est_par.var_par.iconst = 1;    % include constant
est_par.var_par.icomp  = 1;    % compute companion form of model .. excluding constant

% yit equation parameters
est_par.n_uarlag = n_lags_uar;  % number of arlags for uniqueness

% Matrices for storing results
n_series = size(datain.bpdata,2);

%% Set the argument values for factor_estimation_ls_full

data = datain.bpdata;
inclcode = datain.bpinclcode;


%% Content from factor_estimation_ls_full.m

% PRELIMINARIES
n_series = size(data,2);                    % number of series
nfirst   = est_par.smpl_par.nfirst;         % start date
nlast    = est_par.smpl_par.nlast;          % end date
calvec   = est_par.smpl_par.calvec;         % calendar
nper     = est_par.smpl_par.nper;           % number of periods a year
n_uarlag = est_par.n_uarlag;                % number of AR lags
ntmin    = est_par.lambda.nt_min;           % minimum number of Obs

% USE SUBSET OF DATA TO ESTIMATE FACTORS
est_data = data(:,inclcode==1);
if levels
    est_data = [nan(1,size(est_data,2)); diff(est_data,1,1)]; % If data is in levels, estimate factors off first differences
end


%% Content from factor_estimation_ls

%%
% Preliminaries
% extract estimation parameters
smpl_par           = est_par.smpl_par;
lambda_constraints = est_par.fac_par.lambda_constraints_est;
nt_min             = est_par.fac_par.nt_min;
tol                = est_par.fac_par.tol;
nfac_u             = est_par.fac_par.nfac.unobserved;
nfac_o             = est_par.fac_par.nfac.observed;
nfac_t             = est_par.fac_par.nfac.total;
if nfac_o > 0;
  w = est_par.fac_par.w;
end;

% Estimate factors with unbalanced panel by LS -- standardize data first
  % Sample period
  [istart, iend] = smpl_HO(smpl_par);
  istart = max(istart,1);
  iend = min(iend,size(est_data,1));

  % Estimate Factors 
  xdata = est_data(istart:iend,:);
  nt = size(xdata,1);
  ns = size(xdata,2);

  % Mean and Standard Deviation
  xmean = nanmean(xdata)';                                  % mean (ignoring NaN)
  mult = sqrt((sum(~isnan(xdata))-1)./sum(~isnan(xdata)));  % num of non-NaN entries for each series
  xstd = (nanstd(xdata).*mult)';                            % std (ignoring NaN)
  xdata_std = (xdata - repmat(xmean',nt,1))./repmat(xstd',nt,1);   % standardized data

  if nfac_o > 0;
      wdata = w(istart:iend,:);
      if sum(sum(isnan(wdata))) ~= 0;
          error('w contains missing values over sample period, processing stops');
      end;
  end;

  n_lc = 0;            % number of constrains placed on lambda
  if size(lambda_constraints,2) > 1;
     lam_c_index = lambda_constraints(:,1);     % Which row of lambda: Constraints are then R*lambda = r
     lam_c_R = lambda_constraints(:,2:end-1);   % R matrix
     lam_c_r = lambda_constraints(:,end);       % r value
     lam_c_r_scl = lam_c_r./xstd(lam_c_index);  % Adjusted for scaling
     n_lc = size(lambda_constraints,1);
  end;

  % Compute Total Sum of Squares
  tss = 0;
  nobs = 0;
  for is = 1:ns;
      tmp = xdata_std(:,is);     % select series
      tmp = tmp(isnan(tmp)==0);  % drop NaN
      tss = tss+sum(tmp.^2);     % add to tss
      nobs = nobs+size(tmp,1);   % add to n*T
  end;

  % Estimate factors using balanced panel
  if nfac_u > 0;
   xbal = packr(xdata_std')';
   %[coef,score,latent]=princomp(xbal);
   [coef,score,latent] = pca(xbal);
   f = score(:,1:nfac_u);
   fa = f;
   if nfac_o > 0;
      fa = [wdata f];
   end;
   lambda = NaN*zeros(ns,nfac_t);
  else;
   fa = wdata;
  end;

  diff = 100;
  ssr = 0;
  while diff>tol*(nt*ns)
      ssr_old = ssr;
    for i = 1:ns; 
        tmp=packr([xdata_std(:,i) fa]);
        if size(tmp,1) >= nt_min;
  	       y=tmp(:,1);
     	     x=tmp(:,2:end);
           xxi = inv(x'*x);
           bols = xxi*(x'*y);
           b = bols;
           if n_lc > 0;
             % Check for restrictions and impose;
             ii = lam_c_index == i;
             if sum(ii) > 0;
                 R = lam_c_R(ii==1,:);
                 r_scl = lam_c_r_scl(ii==1,:);
                 tmp1 = xxi*R';
                 tmp2 = inv(R*tmp1);
                 b = bols - tmp1*tmp2*(R*bols-r_scl);
             end;
           end;
           lambda(i,:)= b';
        end;
    end;
    edata = xdata_std;
    if nfac_u > 0;
       if nfac_o > 0;
          edata = xdata_std - fa(:,1:nfac_o)*lambda(:,1:nfac_o)';
       end;
       for t = 1:nt;
          tmp=packr([edata(t,:)' lambda(:,nfac_o+1:end)]);
          y=tmp(:,1);
     	    x=tmp(:,2:end);
          b = x\y;
          f(t,:) = b';
       end; 
       fa = f;
       if nfac_o > 0;
         fa = [wdata f];
       end;
    end;
    % Compute residuals 
    e = xdata_std-fa*lambda';
    ssr = sum(nansum(e.^2));
    diff = abs(ssr_old - ssr);
  end;
  
  f_est = fa;

  % Compute R2 for each series
  r2vec = NaN*zeros(ns,1);
  for i = 1:ns;
      tmp=packr([xdata_std(:,i) f_est]);
      if size(tmp,1) >= nt_min;
  	    y=tmp(:,1);
        x=tmp(:,2:end);
        b = x\y;
        e = y -x*b;
        r2_ssr = e'*e;
        r2_tss = y'*y;
        r2vec(i) = 1-r2_ssr/r2_tss;
      end;
  end;

  fac_est = NaN(size(est_data,1),nfac_t);
  fac_est(istart:iend,:)=f_est;
  lambda = lambda.*repmat(xstd,1,nfac_t);

  lsout.fac = fac_est;   lsout.lambda = lambda;
  lsout.tss = tss;       lsout.ssr = ssr;
  lsout.r2vec = r2vec;   lsout.nobs = nobs;
  lsout.nt = nt;         lsout.ns = ns;

%% Save the data
save("../lpw_est_data.mat", 'est_data')

