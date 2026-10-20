from PIL import Image

img = Image.open('PAGAMENTO.png')
w, h = img.size

cols = 2
rows = 3

for r in range(rows):
    for c in range(cols):
        left = round(c * w / cols)
        right = round((c + 1) * w / cols)
        top = round(r * h / rows)
        bottom = round((r + 1) * h / rows)
        
        tile = img.crop((left, top, right, bottom))
        tile.save(f'folha_L{r+1}_C{c+1}.png')