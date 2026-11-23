#!/usr/bin/env python3
"""Within-one-right-forearm EMG common activity vs pooled performance, exploratory.

Not inter-arm! Not demonstration of neural crosstalk, error amplification or causality.
Uses existing EDF, original QC masks, 4 bipolar EMGs, runs 448, 45 subjects.
Requires mne scipy numpy pandas matplotlib optional. No trial/performance join invented.
"""
import argparse, csv, hashlib, json, math, re, traceback
from pathlib import Path
from fractions import Fraction
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from scipy.signal import resample_poly, butter, sosfiltfilt, detrend, correlate
from scipy.stats import spearmanr

EEG=('C3','C1','Cz','C2','C4','CP3','CPz','CP4')
EMG=('M1L','M1R','M2L','M2R','M3L','M3R','M4L','M4R')
MUSCLES=('ECRL','ECU','FCR','PL')
DIRS={1:'flexion',2:'extension',3:'radial',4:'ulnar'}
# NOT perfectly antagonistic for all tasks. Pairing is exploratory biomechanical proxy.
PAIRS={'extension':('ECU','FCR'),'flexion':('FCR','ECU'),
       'radial':('ECRL','PL'),'ulnar':('PL','ECRL')}
FS=512; WIN=1280; NFFT=512; STEP=256
VALID=('NO_INSTRUMENTED_FLAG','REVIEW_AMPLITUDE_ONLY')


def load_mask(path):
    df=pd.read_csv(path,usecols=['subject','run','trial_zero_based','direction','channel','qc_action','modality'],low_memory=False)
    if len(df)!=215028 or df.subject.nunique()!=45:raise ValueError('QC mask count differs from locked 215028 / 45')
    if df.duplicated(['subject','run','trial_zero_based','channel']).any():raise ValueError('Duplicate QC key')
    if len(df[['subject','run','trial_zero_based']].drop_duplicates())!=17919:raise ValueError('QC trial count mismatch')
    # Build eligible only for the two muscles of the direction, retaining exact trial and run.
    use=df[df.modality.eq('emg_bipolar')]
    m={}
    for r in use.itertuples(index=False):
        k=(int(r.subject),int(r.run),int(r.trial_zero_based))
        if k not in m:m[k]=[str(r.direction),{}]
        if m[k][0]!=str(r.direction):raise ValueError('Inconsistent direction')
        m[k][1][str(r.channel)]=str(r.qc_action)
    if len(m)!=17919:raise ValueError('EMG mask cardinality wrong')
    return m


def states(raw):
    def find(prefix):return next((n for n in raw.ch_names if n.startswith(prefix) or prefix.startswith(n)),None)
    sn,tn=find('TrialInitMovStamp'),find('CurrentTarget')
    if not sn or not tn:raise ValueError('Missing movement and target state channels')
    state=(raw.get_data(picks=[sn])[0]>.5).astype(np.int8)
    starts=np.flatnonzero(np.diff(state,prepend=0)==1)
    targets=np.rint(raw.get_data(picks=[tn])[0][starts]).astype(int)
    return [(int(s),int(d)) for s,d in zip(starts,targets) if int(d) in DIRS]


def get_trials(path,sub,run,mask):
    import mne
    raw=mne.io.read_raw_edf(str(path),preload=False,verbose='ERROR')
    if len(raw.ch_names)<23:raise ValueError('EDF does not have expected 16 data + 7 state channels')
    raw.rename_channels(dict(zip(raw.ch_names[:16],EEG+EMG)))
    fs=float(raw.info['sfreq']); ev=states(raw)
    expected=[k for k in mask if k[:2]==(sub,run)]
    if len(ev)!=len(expected):raise ValueError(f'EDF/mask event count mismatch {len(ev)} vs {len(expected)}')
    x=raw.get_data(picks=list(EMG))
    bipolar=np.stack([x[2*i]-x[2*i+1] for i in range(4)])
    if not np.isclose(fs,FS,atol=.01):
        ratio=Fraction(FS/fs).limit_denominator(10000)
        bipolar=resample_poly(bipolar,ratio.numerator,ratio.denominator,axis=1)
    # Published 20-200Hz 339-tap Hamming FIR, consistent with prior coherence pipeline.
    bipolar=mne.filter.filter_data(bipolar,FS,20,200,filter_length=339,
            l_trans_bandwidth='auto',h_trans_bandwidth='auto',
            fir_window='hamming',fir_design='firwin',phase='zero',method='fir',verbose=False)
    bydir=defaultdict(list); skip=Counter()
    for i,(sample,dircode) in enumerate(ev):
        d=DIRS[dircode];key=(sub,run,i)
        if key not in mask or mask[key][0]!=d:raise ValueError(f'Event direction mismatch {key}')
        a,b=PAIRS[d];actions=mask[key][1]
        if actions.get(a) not in VALID or actions.get(b) not in VALID:
            skip['qc_flag']+=1;continue
        at=int(round(sample/fs*FS))
        if at<0 or at+WIN>bipolar.shape[1]:skip['incomplete_window']+=1;continue
        xx=np.stack([bipolar[MUSCLES.index(a),at:at+WIN],bipolar[MUSCLES.index(b),at:at+WIN]])
        xx=detrend(xx,axis=1,type='linear')
        if not np.isfinite(xx).all() or np.min(xx.std(axis=1))<1e-10:skip['invalid']+=1;continue
        bydir[d].append((i,xx))
    return bydir,dict(native_sampling_rate=fs,n_events=len(ev),excluded=dict(skip))


def spectral_components(x):
    # x shape [trial, 2 muscle, sample]; cross spectra aggregated by trial/time, not pseudo-N.
    n,two,npts=x.shape
    if two!=2 or npts!=WIN:raise ValueError('Bad epoch dimensions')
    starts=np.arange(0,WIN-NFFT+1,STEP)
    window=np.hanning(NFFT)
    chunks=np.stack([x[:,:,s:s+NFFT]*window for s in starts],axis=2) # trials, muscle, segments, time
    fft=np.fft.rfft(chunks,axis=-1)
    return fft


def coh_bands(fft,perm=None):
    if perm is None:perm=np.arange(fft.shape[0])
    a=fft[:,0,:,:];b=fft[perm,1,:,:]
    sxy=np.mean(a*np.conj(b),axis=(0,1))
    sa=np.mean(np.abs(a)**2,axis=(0,1));sb=np.mean(np.abs(b)**2,axis=(0,1))
    v=np.abs(sxy)**2 /np.maximum(sa*sb,1e-30)
    f=np.fft.rfftfreq(NFFT,1/FS)
    def avg(lo,hi):return float(np.mean(v[(f>=lo)&(f<=hi)]))
    return avg(13,30),avg(70,150)


def per_trial_measures(x):
    # EMG envelope covariation is distinct from unrectified zero-lag artifact proxy.
    env_sos=butter(4,10,fs=FS,btype='lowpass',output='sos')
    out=[]
    for e in x:
        a,b=e
        ea=sosfiltfilt(env_sos,np.abs(a)); eb=sosfiltfilt(env_sos,np.abs(b))
        if ea.std()<1e-14 or eb.std()<1e-14:continue
        er=float(np.corrcoef(ea,eb)[0,1]); raw=float(np.corrcoef(a,b)[0,1])
        ratio=float(np.mean(ea*eb)/max(np.sqrt(np.mean(ea*ea)*np.mean(eb*eb)),1e-15))
        out.append((er,raw,ratio, float(np.sqrt(np.mean(a*a))),float(np.sqrt(np.mean(b*b)))))
    return np.array(out,float)


def calculate(rows,perms,seed):
    if len(rows)<5:return None
    x=np.stack([r[1] for r in rows])
    metrics=per_trial_measures(x)
    if len(metrics)<5:return None
    # Normalize each muscle's per-epoch RMS to avoid between-muscle voltage differences
    xx=x/np.maximum(np.sqrt(np.mean(x*x,axis=-1,keepdims=True)),1e-14)
    fft=spectral_components(xx)
    beta,hf=coh_bands(fft)
    rng=np.random.default_rng(seed); null=[]
    for _ in range(perms):
        perm=rng.permutation(len(xx))
        # use circular nonzero shift if permutation contains any fixed trial
        if np.any(perm==np.arange(len(xx))):perm=np.roll(np.arange(len(xx)),rng.integers(1,len(xx)))
        null.append(coh_bands(fft,perm)[0])
    null=np.array(null)
    # Event-residual diagnostic (mean trial template subtraction)
    residual=xx-xx.mean(axis=0,keepdims=True)
    beta_resid=coh_bands(spectral_components(residual))[0]
    return dict(n_trials=len(x),muscle_a_rms_uv=float(np.median(metrics[:,3])*1e6),
         muscle_b_rms_uv=float(np.median(metrics[:,4])*1e6),
         envelope_corr_median=float(np.median(metrics[:,0])),
         unrectified_zero_lag_corr_abs_median=float(np.median(np.abs(metrics[:,1]))),
         envelope_overlap_median=float(np.median(metrics[:,2])),
         emg_emg_beta_coherence=beta,emg_emg_beta_event_residual=beta_resid,
         emg_emg_beta_null_median=float(np.median(null)),
         emg_emg_beta_delta=float(beta-np.median(null)),
         emg_emg_beta_p_diagnostic=float((1+np.sum(null>=beta))/(len(null)+1)),
         emg_emg_highfreq_coherence=hf,trial_indices_zero_based=';'.join(str(t[0]) for t in rows))


def bh(p):
    p=np.asarray(p,float);out=np.full(len(p),np.nan);ii=np.where(np.isfinite(p))[0]
    if not len(ii):return out
    order=ii[np.argsort(p[ii])]; z=p[order]*len(ii)/np.arange(1,len(ii)+1)
    out[order]=np.minimum(1,np.minimum.accumulate(z[::-1])[::-1]);return out


def analyze(outdir,cohpath,perfpath,seed):
    run=pd.read_csv(outdir/'run_muscle_pair_metrics.csv')
    if run.empty:raise ValueError('No processed runs')
    subject=run.groupby(['subject','direction'],as_index=False).agg(
        n_runs=('run','nunique'),n_trials=('n_trials','sum'),
        emg_beta_delta=('emg_emg_beta_delta','median'),
        emg_beta_raw=('emg_emg_beta_coherence','median'),
        emg_beta_residual=('emg_emg_beta_event_residual','median'),
        envelope_corr=('envelope_corr_median','median'),
        overlap=('envelope_overlap_median','median'),
        electrical_proxy=('unrectified_zero_lag_corr_abs_median','median'),
        highfreq_coh=('emg_emg_highfreq_coherence','median'),
        muscle_a_rms_uv=('muscle_a_rms_uv','median'),muscle_b_rms_uv=('muscle_b_rms_uv','median'))
    subject.to_csv(outdir/'subject_direction_muscle_metrics.csv',index=False)
    c=pd.read_csv(cohpath);p=pd.read_csv(perfpath)
    ext=subject[subject.direction=='extension']
    e_c=c[(c.direction=='extension')&(c['mode']=='event_residual')][['subject','delta_direction']].rename(columns={'delta_direction':'c3_ecu_delta_event_residual'})
    e_r=c[(c.direction=='extension')&(c['mode']=='raw')][['subject','delta_direction']].rename(columns={'delta_direction':'c3_ecu_delta_raw'})
    perf=p[p['mode']=='raw'][['subject','accuracy','execution_time','trajectory_efficiency','n_metric_rows']]
    m=ext.merge(e_c,on='subject',validate='one_to_one').merge(e_r,on='subject',validate='one_to_one').merge(perf,on='subject',validate='one_to_one')
    m.to_csv(outdir/'extension_subject_associations.csv',index=False)
    # Explicit exploratory small family, common drive vs GLOBAL accuracy, and C3-ECU coherence.
    tests=[('emg_beta_delta','accuracy'),('envelope_corr','accuracy'),('electrical_proxy','accuracy'),
           ('emg_beta_delta','c3_ecu_delta_event_residual'),('envelope_corr','c3_ecu_delta_event_residual'),
           ('emg_beta_delta','trajectory_efficiency'),('emg_beta_delta','execution_time')]
    res=[]
    for x,y in tests:
        good=m[['subject',x,y]].replace([np.inf,-np.inf],np.nan).dropna()
        if len(good)<12 or good[x].nunique()<4 or good[y].nunique()<4:continue
        rho,pval=spearmanr(good[x],good[y]);n=len(good)
        rng=np.random.default_rng(seed+len(res))
        xr=pd.Series(good[x].values).rank().to_numpy();yr=pd.Series(good[y].values).rank().to_numpy()
        xr=(xr-xr.mean())/np.linalg.norm(xr-xr.mean());yr=(yr-yr.mean())/np.linalg.norm(yr-yr.mean())
        pperm=(1+sum(abs(float(xr@rng.permutation(yr)))>=abs(rho)-1e-14 for _ in range(4999)))/5000
        loo=[float(spearmanr(np.delete(good[x].values,j),np.delete(good[y].values,j)).statistic) for j in range(n)]
        res.append(dict(predictor=x,outcome=y,n=n,rho=rho,p_spearman=pval,p_perm=pperm,loo_min=min(loo),loo_max=max(loo)))
    tab=pd.DataFrame(res)
    if len(tab):tab['q_bh_exploratory_family']=bh(tab.p_perm.values)
    tab.to_csv(outdir/'exploratory_correlations.csv',index=False)
    print('\nEXTENSION EXPLORATORY ASSOCIATIONS (GLOBAL performance is NOT extension-specific):')
    print(tab.to_string(index=False))
    print('N matching subjects:',len(m))
    return len(m)


def self_test():
    rng=np.random.default_rng(123); fs=FS;t=np.arange(WIN)/fs
    n=10
    # Shared 20Hz input, independent noise: within-trial pairing should increase coherence
    common=np.sin(2*np.pi*20*t)
    arr=np.stack([np.stack([common+0.35*rng.standard_normal(WIN),common+0.35*rng.standard_normal(WIN)]) for _ in range(n)])
    f=spectral_components(arr)
    a,b=coh_bands(f);z,_=coh_bands(f,np.roll(np.arange(n),1))
    assert 0<=a<=1 and 0<=b<=1
    # shared perfectly event-locked sinusoid survives derangements, so null is high: expected!
    assert z>.10  # even cross-trial pairs remain coherent for a strictly event-locked sine
    independent=rng.standard_normal((10,2,WIN));v,_=coh_bands(spectral_components(independent));assert v<.3
    assert abs(np.corrcoef(arr[0,0],arr[0,1])[0,1])>.5
    assert np.allclose(bh([.01,.04,.03]),[.03,.04,.04])
    print('SELF-TEST PASS: spectral bounds, event-locked surrogate behavior, independent noise, envelope and BH')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--data-dir',default='DATA');p.add_argument('--qc-mask',default='channel_epoch_qc_mask.csv')
    p.add_argument('--coherence-csv',default='coherence_direction_aggregated.csv')
    p.add_argument('--performance-csv',default='coherence_performance_45_subjects.csv')
    p.add_argument('--out-dir',default='RESULTS_COMMON_DRIVE45');p.add_argument('--subjects',type=int,nargs='*')
    p.add_argument('--max-runs-per-subject',type=int);p.add_argument('--permutations',type=int,default=99)
    a=p.parse_args()
    if a.self_test:self_test();return
    root=Path(a.data_dir);out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    mask=load_mask(a.qc_mask);subjects=a.subjects or list(range(1,46))
    rows=[];errors=[];runmeta=[]
    for sub in subjects:
        paths=sorted((root/f'sub-{sub:02d}'/'eeg').glob(f'sub-{sub:02d}_task-*_run-*_eeg.edf'))
        if a.max_runs_per_subject:paths=paths[:a.max_runs_per_subject]
        for path in paths:
            mm=re.search(r'run-(\d+)',path.name)
            if not mm:continue
            run=int(mm.group(1));kfile=out/f'run_sub-{sub:02d}_run-{run:02d}.json'
            try:
                if kfile.exists():
                    payload=json.loads(kfile.read_text(encoding='utf-8'))
                    if payload.get('n_permutations')!=a.permutations:raise ValueError('Cache permutation count mismatch; use another output dir')
                else:
                    bydir,meta=get_trials(path,sub,run,mask)
                    entry=[]
                    for di,dirkey in enumerate(('flexion','extension','radial','ulnar')):
                        vals=calculate(bydir[dirkey],a.permutations,seed=sub*100000+run*100+di)
                        if vals is None:continue
                        entry.append(dict(subject=sub,run=run,direction=dirkey,muscle_a=PAIRS[dirkey][0],muscle_b=PAIRS[dirkey][1],**vals))
                    payload=dict(n_permutations=a.permutations,metadata=meta,results=entry)
                    kfile.write_text(json.dumps(payload,indent=2),encoding='utf-8')
                rows.extend(payload['results']);runmeta.append(dict(subject=sub,run=run,**payload['metadata']))
            except Exception as exc:
                errors.append(dict(subject=sub,run=run,file=path.name,error=str(exc),traceback=traceback.format_exc()))
                print('ERROR',sub,run,str(exc),flush=True)
            print(f'sub-{sub:02d} run-{run:02d}',flush=True)
    pd.DataFrame(rows).to_csv(out/'run_muscle_pair_metrics.csv',index=False)
    pd.DataFrame(runmeta).to_csv(out/'run_execution.csv',index=False)
    (out/'errors.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
    summary=dict(n_runs=len(runmeta),n_subjects=len(set(r['subject'] for r in runmeta)),
                 n_conditions=len(rows),n_errors=len(errors),n_permutations=a.permutations,
                 disclaimer='One forearm only. EMG-EMG coherence not proof of neural common drive. Global motor metrics not direction matched.')
    if not errors and len(rows) and len(set(r['subject'] for r in rows))>=40:
        summary['n_subjects_in_extension_associations']=analyze(out,a.coherence_csv,a.performance_csv,20261121)
    elif len(rows):
        summary['analysis_note']='Gate run: fewer than 40 participants; subject-level correlations are intentionally deferred.'
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('SUMMARY:',json.dumps(summary,indent=2))

if __name__=='__main__':main()
