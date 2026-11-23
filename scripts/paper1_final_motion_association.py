#!/usr/bin/env python3
"""Fixed-scope, run-matched movement endpoint CONSISTENCY analysis.

No epoch join to MotorPerformanceMetrics.tsv. This proxy is NOT target accuracy.
Motion/EDF timing is validated before statistical estimation. Does not claim causation.
"""
import argparse, json, re, traceback, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DIRECTIONS={1:'flexion',2:'extension',3:'radial',4:'ulnar'}
PREDICTORS=['envelope_corr_median','emg_emg_beta_delta','unrectified_zero_lag_corr_abs_median']


def eeg_events(edf):
    import mne
    raw=mne.io.read_raw_edf(str(edf),preload=False,verbose='ERROR')
    def get(name):
        found=[c for c in raw.ch_names if c.startswith(name) or name.startswith(c)]
        if len(found)!=1:raise ValueError(f'State channel {name}: found {found}')
        return raw.get_data(picks=[found[0]])[0]
    state=(get('TrialInitMovStamp')>.5).astype(np.int8)
    starts=np.flatnonzero(np.diff(state,prepend=0)==1)
    targets=np.rint(get('CurrentTarget')[starts]).astype(int)
    return float(raw.info['sfreq']),float(raw.n_times/raw.info['sfreq']),[(int(s),DIRECTIONS.get(int(d))) for s,d in zip(starts,targets)]


def parse_motion(path):
    # All rows are numeric data, not column names.
    arr=pd.read_csv(path,sep='\t',header=None,dtype='float64').to_numpy()
    if arr.ndim!=2 or arr.shape[1]!=2:raise ValueError(f'Expected 2 numeric motion columns, got {arr.shape}')
    if not np.isfinite(arr).all():raise ValueError('Motion has nonfinite values')
    return arr


def endpoint_data(arr,events,edf_fs,fs_motion):
    result=[]
    n=len(arr)
    for ti,(s,dr) in enumerate(events):
        if dr not in DIRECTIONS.values():continue
        t=s/edf_fs
        # Movement state marks the onset. Compare median 150ms after onset and last 200ms of 2.5s epoch.
        a=int(round((t+.05)*fs_motion));b=int(round((t+.20)*fs_motion))
        c=int(round((t+2.25)*fs_motion));d=int(round((t+2.45)*fs_motion))
        z0=int(round((t-.80)*fs_motion));z1=int(round((t-.20)*fs_motion))
        if z0<0 or d>n or b<=a or d<=c:continue
        baseline=np.median(arr[a:b,:],axis=0)
        endpoint=np.median(arr[c:d,:],axis=0)
        # Dynamic response and pre-event noise as diagnostic only.
        mov=arr[a:d,:];pre=arr[z0:z1,:]
        dynamic=float(np.linalg.norm(np.percentile(mov,90,axis=0)-np.percentile(mov,10,axis=0)))
        pren=float(np.linalg.norm(np.percentile(pre,90,axis=0)-np.percentile(pre,10,axis=0)))
        v=endpoint-baseline
        result.append(dict(trial_zero_based=ti,direction=dr,start_x=float(baseline[0]),start_y=float(baseline[1]),
                           dx=float(v[0]),dy=float(v[1]),excursion=float(np.linalg.norm(v)),
                           motion_dynamic=dynamic,pre_dynamic=pren,
                           active=bool(dynamic>max(2.,3.*pren)),valid=bool(np.all(np.isfinite(v)))))
    return pd.DataFrame(result)


def summarize_by_run(trials,sub,run):
    output=[]
    for direction,g in trials.groupby('direction'):
        valid=g[g.active & g.valid]
        if len(valid)<5:continue
        v=valid[['dx','dy']].to_numpy(float)
        center=np.median(v,axis=0)
        norm=float(np.linalg.norm(center))
        # Do not label displacement variability as error to target.
        meddev=float(np.median(np.linalg.norm(v-center,axis=1)))
        output.append(dict(subject=sub,run=run,direction=direction,n_motion_trials=len(g),n_valid_trials=len(valid),
             active_fraction=float(g.active.mean()),median_displacement_counts=norm,
             endpoint_variability_counts=meddev,
             endpoint_variability_normalized=meddev/max(norm,1.),
             start_x_median=float(np.median(valid.start_x)),start_y_median=float(np.median(valid.start_y))))
    return output


def holo(p):
    ps=np.asarray(p,float);n=len(ps);order=np.argsort(ps);q=np.empty(n)
    q[order]=np.minimum(1,np.maximum.accumulate(ps[order]*(n-np.arange(n))))
    return q


def stats(rows,folder):
    import statsmodels.api as sm
    folder=Path(folder)
    df=rows[(rows.direction=='extension')].copy()
    df=df.replace([np.inf,-np.inf],np.nan)
    ycol='endpoint_variability_normalized'
    selected=['subject','run','envelope_corr_median','emg_emg_beta_delta','unrectified_zero_lag_corr_abs_median',
             'muscle_a_rms_uv','muscle_b_rms_uv',ycol]
    df=df.dropna(subset=selected)
    df=df[(df[ycol]>=0)&(df[ycol]<100)]
    if df.subject.nunique()<30 or len(df)<150:
        raise RuntimeError(f'INSUFFICIENT MATCHED DATA for analysis: {df.subject.nunique()} participants, {len(df)} runs. Inspect sync and missingness.')
    # Exactly 3 predeclared subject-level correlations, Holm over three.
    sub=df.groupby('subject',as_index=False).agg({ycol:'median',
         'envelope_corr_median':'median','emg_emg_beta_delta':'median',
         'unrectified_zero_lag_corr_abs_median':'median'})
    testing=[]
    for pred in PREDICTORS:
        rho,p=spearmanr(sub[pred],sub[ycol])
        testing.append(dict(predictor=pred,outcome=ycol,n_subjects=len(sub),rho_spearman=float(rho),p_two_sided=float(p)))
    tab=pd.DataFrame(testing);tab['p_holm_three_tests']=holo(tab.p_two_sided.values)
    tab.to_csv(folder/'PRIMARY_subject_correlations.csv',index=False)
    sub.to_csv(folder/'extension_subject_level.csv',index=False)
    # Main run-level within-person regression, dependent outcome is log1p variability proxy.
    # Fixed subject demeaning, SE clustered by participant. Exact same direction in every run.
    df['outcome_log']=np.log1p(df[ycol])
    for x in ['envelope_corr_median','unrectified_zero_lag_corr_abs_median',
              'muscle_a_rms_uv','muscle_b_rms_uv','outcome_log']:
        if x.endswith('_rms_uv'):df['log_'+x]=np.log1p(df[x].clip(lower=0))
    # Predictor is standardized within subject. Covariates are also within centered.
    factors=['envelope_corr_median','unrectified_zero_lag_corr_abs_median','log_muscle_a_rms_uv','log_muscle_b_rms_uv']
    for f in factors+['outcome_log']:
        df[f+'_within']=df[f]-df.groupby('subject')[f].transform('mean')
    X=df[[f+'_within' for f in factors]].to_numpy(float)
    y=df['outcome_log_within'].to_numpy(float)
    sx=np.std(X,axis=0)
    if np.any(sx<1e-9):raise RuntimeError('Zero within-person predictor variance')
    X=X/sx
    mod=sm.OLS(y,X).fit(cov_type='cluster',cov_kwds={'groups':df.subject.values})
    model=pd.DataFrame({'predictor':factors,'coef_per_1SD_within':mod.params,
         'se_clustered':mod.bse,'p_clustered':mod.pvalues,
         'ci_low':mod.conf_int()[:,0],'ci_high':mod.conf_int()[:,1]})
    model.to_csv(folder/'SECONDARY_within_subject_regression.csv',index=False)
    with open(folder/'statistical_summary.json','w',encoding='utf-8') as f:
        json.dump(dict(n_subjects=int(df.subject.nunique()),n_runs=int(len(df)),
           primary='Three subject-level extension correlations, Holm-adjusted',
           secondary='Within-participant de-meaned run-level regression, clustered SE',
           crucial='Endpoint DISPERSION around per-run direction median, not actual target error; no causal attribution.'),f,indent=2)
    print('PRIMARY SUBJECT CORRELATIONS\n',tab.to_string(index=False))
    print('SECONDARY WITHIN SUBJECT\n',model.to_string(index=False))
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        for pred in PREDICTORS:
            fig,ax=plt.subplots(figsize=(6.5,4.8))
            ax.scatter(sub[pred],sub[ycol],alpha=.75)
            ax.set(xlabel=pred,ylabel='Normalized endpoint dispersion (NOT target error)',title='Extension: n='+str(len(sub)))
            fig.tight_layout();fig.savefig(folder/('scatter_'+pred+'.png'),dpi=160);plt.close(fig)
    except ImportError:pass


def run(a):
    root=Path(a.data_dir);out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    common=pd.read_csv(a.common_drive_csv);cmc=pd.read_csv(a.cmc_csv)
    if common.duplicated(['subject','run','direction']).any():raise ValueError('Duplicate common-drive rows')
    cmc=cmc[(cmc['mode']=='event_residual') & (cmc['direction']=='extension')]
    cmc=cmc[['subject','run','direction','delta']].rename(columns={'delta':'c3_ecu_event_residual_delta'})
    if cmc.duplicated(['subject','run','direction']).any():raise ValueError('Duplicate CMC rows')
    selected=a.subjects or sorted(int(v) for v in common.subject.unique())
    recorded=[];audits=[];fail=[];alltrials=[]
    for sub in selected:
        runvals=sorted(common.loc[common.subject==sub,'run'].unique())
        if a.max_runs:runvals=runvals[:a.max_runs]
        for run in runvals:
            motion=root/f'sub-{sub:02d}'/'motion'/f'sub-{sub:02d}_task-WristPointingTask_tracksys-BiomechWrist_run-{run:02d}_motion.tsv'
            edf=root/f'sub-{sub:02d}'/'eeg'/f'sub-{sub:02d}_task-WristPointingTask_run-{run:02d}_eeg.edf'
            try:
                if not motion.exists() or not edf.exists():raise FileNotFoundError(f'Missing paired file: motion={motion.exists()} edf={edf.exists()}')
                fs,duration,ev=eeg_events(edf)
                arr=parse_motion(motion)
                if len(ev) not in (39,40):raise ValueError(f'Unexpected number of events {len(ev)}')
                fs_motion=len(arr)/duration
                if not 500<=fs_motion<=524:raise ValueError(f'Motion sample rate/duration mismatch: {fs_motion:.4f} Hz (nominal 512)')
                trials=endpoint_data(arr,ev,fs,fs_motion)
                if len(trials)<38:raise ValueError(f'Too few motion epochs {len(trials)}')
                activity=float(trials.active.mean())
                # Require objective evidence of time-locked motion to guard false alignment.
                if activity<.60:raise ValueError(f'FAIL motion-EEG alignment gate: {activity:.2%} movement epochs active')
                res=summarize_by_run(trials,sub,int(run))
                if len(res)<3:raise ValueError(f'Only {len(res)} directions with >=5 active trials')
                recorded.extend(res)
                if a.save_epochs:
                    for r in trials.to_dict('records'):alltrials.append(dict(subject=sub,run=int(run),**r))
                audits.append(dict(subject=sub,run=int(run),n_events=len(ev),n_motion_rows=len(arr),
                   implied_motion_hz=fs_motion,active_fraction=activity,
                   n_valid_directions=len(res),status='PASS'))
            except Exception as e:
                fail.append(dict(subject=sub,run=int(run),error=str(e)))
                audits.append(dict(subject=sub,run=int(run),status='FAIL',reason=str(e)))
            print(f'sub-{sub:02d} run-{int(run):02d}',audits[-1]['status'],flush=True)
    audit=pd.DataFrame(audits);audit.to_csv(out/'run_alignment_audit.csv',index=False)
    pd.DataFrame(fail,columns=['subject','run','error']).to_csv(out/'run_failures.csv',index=False)
    if a.save_epochs and alltrials:pd.DataFrame(alltrials).to_csv(out/'movement_epoch_diagnostics.csv',index=False)
    merged=pd.DataFrame(recorded)
    if len(merged):
        merged=merged.merge(common,on=['subject','run','direction'],how='left',validate='one_to_one')
        merged=merged.merge(cmc,on=['subject','run','direction'],how='left',validate='one_to_one')
        merged.to_csv(out/'matched_run_direction.csv',index=False)
    summary=dict(n_subjects_requested=len(selected),runs_checked=len(audits),runs_passed=int((audit.status=='PASS').sum()),
         n_failed=len(fail),n_matched=int(len(merged)),
         warning='Not target accuracy. Direction-matched per-run endpoint dispersion (encoder units) only. Reject unreliable synchronization.')
    (out/'alignment_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('ALIGNMENT SUMMARY:',json.dumps(summary,indent=2))
    if a.gate_only:return
    # Conservative full-data gate: no silent subset cherry-picking.
    if len(selected)!=45 or int((audit.status=='PASS').sum())<350 or len(merged)==0:
        raise RuntimeError('STOP: full-cohort quality gate not satisfied; review run_alignment_audit.csv before inference.')
    stats(merged,out)


def selftest():
    x=np.zeros((512*5,2));fs=512
    x[512:512*4,0]=np.linspace(0,50,512*3)
    e=[(512, 'extension')]
    r=endpoint_data(x,e,fs,fs)
    assert len(r)==1 and r.iloc[0].excursion>10 and bool(r.iloc[0].active)
    z=pd.DataFrame([{'dx':1,'dy':0,'active':True,'valid':True,'direction':'extension','start_x':0,'start_y':0},
                    {'dx':2,'dy':0,'active':True,'valid':True,'direction':'extension','start_x':0,'start_y':0}]*3)
    rr=summarize_by_run(z,1,1);assert rr[0]['n_valid_trials']==6
    assert np.allclose(holo([.01,.04,.03]),[.03,.06,.06])
    print('SELF-TEST PASS: motion row parsing logic, event-window proxy, run aggregation, Holm adjustment')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true');p.add_argument('--gate-only',action='store_true')
    p.add_argument('--data-dir',default='DATA');p.add_argument('--common-drive-csv',default='run_muscle_pair_metrics.csv')
    p.add_argument('--cmc-csv',default='run_direction_coherence.csv');p.add_argument('--out-dir',default='RESULTS_FINAL_MOTION45')
    p.add_argument('--subjects',nargs='+',type=int);p.add_argument('--max-runs',type=int);p.add_argument('--save-epochs',action='store_true')
    a=p.parse_args()
    if a.self_test:selftest()
    else:run(a)
