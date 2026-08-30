import numpy as np
from ..features import extract_v4_features

def test_v4_no_future_feature():
    n=6000; t=np.arange(n)/100.; p=np.sin(t)+100; e=np.sin(t*2)+2
    cycle={"t_abs_s":t,"bladder_pressure_mmHg":p,"cmg_valid_100hz":np.ones(n,bool),"eus_envelope_100hz":e,"eus_valid_100hz":np.ones(n,bool),"eus_raw_native":np.repeat(e,100),"t_eus_abs_native":np.arange(n*100)/10000.,"eus_fs_native":10000.}
    a,_=extract_v4_features(cycle,4000,3800,t[4000]); p[4001:]=1e9; e[4001:]=1e9
    b,_=extract_v4_features(cycle,4000,3800,t[4000]);
    for k in a:
        if isinstance(a[k],float) and np.isfinite(a[k]) and np.isfinite(b.get(k,np.nan)): assert a[k]==b[k]
