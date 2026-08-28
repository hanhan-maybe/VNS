import json, sys
import sonpy as sp
from pathlib import Path

for path in sorted(Path(r"D:\Sparc").glob("STxF*.smrx")):
    f=sp.SonFile(str(path), True)
    rows=[]
    for ch in range(f.MaxChannels()):
        typ=f.ChannelType(ch)
        if int(typ)==0: continue
        rows.append([ch,str(typ).split('.')[-1],f.GetChannelTitle(ch),f.GetChannelUnits(ch),f.GetChannelComment(ch),f.GetIdealRate(ch),f.ChannelDivide(ch)])
    print(json.dumps({"file":path.name,"error":f.GetOpenError(),"tb":f.GetTimeBase(),"duration":f.MaxTime()*f.GetTimeBase(),"channels":rows},ensure_ascii=False))
