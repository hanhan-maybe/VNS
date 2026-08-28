from pathlib import Path
from PIL import Image, ImageDraw

root=Path(r"D:\cubeIDE\project\VNS\data\baseline")
subjects=["STxF14","STxF21","STxF22","STxF23","STxF24","STxF26","STxF27","STxF29","STxF30"]
for name in ("pre_stim_quicklook.png","pre_stim_tail_check.png"):
    thumbs=[]
    for s in subjects:
        im=Image.open(root/s/name).convert("RGB")
        im.thumbnail((600,350))
        canvas=Image.new("RGB",(620,390),"white")
        canvas.paste(im,((620-im.width)//2,25))
        ImageDraw.Draw(canvas).text((10,5),s,fill="black")
        thumbs.append(canvas)
    sheet=Image.new("RGB",(1860,1170),"white")
    for i,im in enumerate(thumbs): sheet.paste(im,((i%3)*620,(i//3)*390))
    sheet.save(Path(r"D:\cubeIDE\project\VNS\.codex_tmp")/("contact_"+name))
