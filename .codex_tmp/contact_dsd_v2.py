from pathlib import Path

from PIL import Image, ImageDraw


root = Path(r"D:\cubeIDE\project\VNS")
for subject in ("STxF21", "STxF26", "STxF27", "STxF29", "STxF30"):
    folder = root / "data" / "dsd_validation" / subject
    for suffix in ("raw_eus_5s", "raw_eus_2s"):
        paths = sorted(folder.glob(f"{subject}_B*_{suffix}.png"))
        images = [Image.open(path).convert("RGB") for path in paths]
        if not images:
            continue
        thumb_width = 1200
        thumbs = []
        for path, source in zip(paths, images):
            height = int(source.height * thumb_width / source.width)
            thumb = source.resize((thumb_width, height))
            canvas = Image.new("RGB", (thumb_width, height + 28), "white")
            canvas.paste(thumb, (0, 28))
            ImageDraw.Draw(canvas).text((5, 5), path.stem, fill="black")
            thumbs.append(canvas)
        width = thumb_width * 2
        height = sum(max(thumbs[i].height, thumbs[i+1].height if i+1 < len(thumbs) else 0)
                     for i in range(0, len(thumbs), 2))
        contact = Image.new("RGB", (width, height), "white")
        y = 0
        for i in range(0, len(thumbs), 2):
            contact.paste(thumbs[i], (0, y))
            row_height = thumbs[i].height
            if i+1 < len(thumbs):
                contact.paste(thumbs[i+1], (thumb_width, y))
                row_height = max(row_height, thumbs[i+1].height)
            y += row_height
        contact.save(root / ".codex_tmp" / f"{subject}_{suffix}_contact.png")
