import json, sys, traceback
import sonpy as sp

path = sys.argv[1]
try:
    f = sp.SonFile(path, True)
except TypeError:
    f = sp.SonFile(path)

out = {
    "open_error": f.GetOpenError(),
    "name": f.GetName(),
    "is64": f.is64file(),
    "version": f.GetVersion(),
    "time_base_s": f.GetTimeBase(),
    "max_channels": f.MaxChannels(),
    "max_time_ticks": f.MaxTime(),
    "file_comments": [],
    "channels": [],
}
for i in range(8):
    try: out["file_comments"].append(f.GetFileComment(i))
    except Exception: break

for ch in range(f.MaxChannels()):
    try:
        typ = f.ChannelType(ch)
        typ_name = str(typ).split('.')[-1]
        if typ_name in ("Off", "None") or int(typ) == 0:
            continue
        rec = {"channel": ch, "type": typ_name}
        for key, meth in [
            ("title", "GetChannelTitle"), ("units", "GetChannelUnits"),
            ("comment", "GetChannelComment"), ("ideal_rate_hz", "GetIdealRate"),
            ("divide_ticks", "ChannelDivide"), ("max_time_ticks", "ChannelMaxTime"),
            ("physical_channel", "PhysicalChannel"), ("scale", "GetChannelScale"),
            ("offset", "GetChannelOffset"), ("item_size", "ItemSize")]:
            try: rec[key] = getattr(f, meth)(ch)
            except Exception as e: rec[key] = f"ERROR: {e}"
        out["channels"].append(rec)
    except Exception as e:
        out["channels"].append({"channel": ch, "error": str(e)})
print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
