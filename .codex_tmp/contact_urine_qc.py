from pathlib import Path
from PIL import Image, ImageDraw

root=Path(r"D:\cubeIDE\project\VNS\data\baseline")
subjects=["STxF14","STxF21","STxF22","STxF23","STxF24","STxF26","STxF27","STxF29","STxF30"]

thumbs=[]
for s in subjects:
    im=Image.open(root/s/"pre_stim_quicklook.png").convert("RGB"); im.thumbnail((600,350))
    c=Image.new("RGB",(620,390),"white"); c.paste(im,((620-im.width)//2,25)); ImageDraw.Draw(c).text((10,5),s,fill="black"); thumbs.append(c)
sheet=Image.new("RGB",(1860,1170),"white")
for i,im in enumerate(thumbs): sheet.paste(im,((i%3)*620,(i//3)*390))
sheet.save(Path(r"D:\cubeIDE\project\VNS\.codex_tmp\urine_quicklooks_contact.png"))

for s in ["STxF14","STxF21","STxF22","STxF23","STxF24","STxF26","STxF27","STxF29","STxF30"]:
    files=sorted((root/s/"void_output_check").glob("*.png"))[:5]
    cards=[]
    for p in files:
        im=Image.open(p).convert("RGB"); im.thumbnail((800,560))
        c=Image.new("RGB",(820,600),"white"); c.paste(im,((820-im.width)//2,25)); ImageDraw.Draw(c).text((10,5),p.stem,fill="black"); cards.append(c)
    sheet=Image.new("RGB",(1640,1800),"white")
    for i,im in enumerate(cards): sheet.paste(im,((i%2)*820,(i//2)*600))
    sheet.save(Path(r"D:\cubeIDE\project\VNS\.codex_tmp")/f"{s}_volume_candidates_contact.png")
