import json, sys, collections, statistics
import sonpy as sp

f=sp.SonFile(sys.argv[1], True)
tb=f.GetTimeBase()
mx=f.MaxTime()
markers=f.ReadMarkers(30, 100000, 0, mx)
events=f.ReadEvents(6, 10000000, 0, mx)

marker_rows=[]
for m in markers:
    codes=[m.Code1,m.Code2,m.Code3,m.Code4]
    marker_rows.append({"time_s":m.Tick*tb,"tick":m.Tick,"codes":codes,
                        "ascii0": chr(codes[0]) if 32 <= codes[0] <= 126 else None})

# pulse trains: split where pulse-to-pulse gap > 1 s
trains=[]
if events:
    start=prev=events[0]; n=1; diffs=[]
    for t in events[1:]:
        dt=(t-prev)*tb
        if dt>1.0:
            trains.append((start,prev,n,diffs))
            start=t;n=1;diffs=[]
        else:
            n+=1;diffs.append(dt)
        prev=t
    trains.append((start,prev,n,diffs))
train_rows=[]
for a,b,n,ds in trains:
    train_rows.append({"start_s":a*tb,"end_s":b*tb,"duration_s":(b-a)*tb,
                       "pulses":n,"median_hz":(1/statistics.median(ds) if ds else None),
                       "prev_marker":max((r for r in marker_rows if r['time_s']<=a*tb),key=lambda x:x['time_s'],default=None)})

out={"duration_s":mx*tb,"event_count":len(events),"marker_count":len(markers),
     "marker_code_counts":{"-".join(map(str,k)):v for k,v in collections.Counter(tuple(r['codes']) for r in marker_rows).items()},
     "markers":marker_rows,"pulse_train_count":len(train_rows),"pulse_trains":train_rows}
print(json.dumps(out,indent=2,ensure_ascii=False))
