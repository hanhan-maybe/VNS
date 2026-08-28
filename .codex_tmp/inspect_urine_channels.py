import json, sys, csv
from pathlib import Path
import numpy as np
import sonpy as sp

root=Path(r"D:\Sparc"); out=Path(r"D:\cubeIDE\project\VNS\data\baseline")
first_map={r["subject"]:float(r["first_stim_s"]) for r in csv.DictReader((out/"pre_stim_inventory.csv").open(encoding="utf-8-sig"))}
for subject in ["STxF14","STxF21","STxF22","STxF23","STxF24","STxF26","STxF27","STxF29","STxF30"]:
    f=sp.SonFile(str(root/f"{subject}.smrx"),True); tb=f.GetTimeBase()
    first=first_map[subject]
    rows=[]
    for ch in range(f.MaxChannels()):
        raw_type=f.ChannelType(ch)
        if int(raw_type)==0: continue
        typ=str(raw_type).split('.')[-1]; title=f.GetChannelTitle(ch); comment=f.GetChannelComment(ch); units=f.GetChannelUnits(ch)
        if typ not in {"Adc","RealWave"} or title not in {"Volume","Leaks","Leak"}: continue
        div=f.ChannelDivide(ch); fs=1/(tb*div); n=int(np.ceil(first*fs)); x=np.asarray(f.ReadFloats(ch,n,0,int(np.ceil(first/tb))),dtype=np.float32)
        p=np.percentile(x[np.isfinite(x)],[0,1,10,50,90,99,100]); dx=np.diff(x.astype(float));
        rows.append({"ch":ch,"title":title,"type":typ,"units":units,"comment":comment,"fs":fs,"n":len(x),"p":p.tolist(),"diff_abs_p99":float(np.percentile(np.abs(dx),99)),"range":float(np.ptp(x))})
    print(json.dumps({"subject":subject,"signals":rows},ensure_ascii=False))
