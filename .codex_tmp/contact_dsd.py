from pathlib import Path
from PIL import Image,ImageDraw
root=Path(r"D:\cubeIDE\project\VNS\data\dsd_validation")
for s in ["STxF21","STxF26","STxF27","STxF29"]:
    for kind in ["eus_zoom","overview"]:
        files=sorted((root/s).glob(f"*_{kind}.png"))
        cards=[]
        for p in files:
            im=Image.open(p).convert("RGB"); im.thumbnail((900,520))
            c=Image.new("RGB",(920,560),"white"); c.paste(im,((920-im.width)//2,25)); ImageDraw.Draw(c).text((10,5),p.stem,fill="black"); cards.append(c)
        sheet=Image.new("RGB",(1840,1680),"white")
        for i,im in enumerate(cards): sheet.paste(im,((i%2)*920,(i//2)*560))
        sheet.save(Path(r"D:\cubeIDE\project\VNS\.codex_tmp")/f"{s}_{kind}_contact.png")
