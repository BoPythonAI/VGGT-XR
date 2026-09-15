"""Create honest fixed-view panels from saved renders; no generated imagery."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont


root=Path('/root/autodl-tmp/vggt-xr');out=root/'outputs/quality/demo';out.mkdir(parents=True,exist_ok=True)
def panel(label,old_run,new_run,indices):
    rows=[]
    for index in indices:
        old=Image.open(old_run/f'comparison_{index:03d}.png').convert('RGB')
        width=old.width//2;gt=old.crop((0,0,width,old.height));baseline=old.crop((width,0,old.width,old.height))
        improved=Image.open(new_run/f'{index:03d}.png').convert('RGB')
        strip=Image.new('RGB',(width*3,old.height+28),'white');strip.paste(gt,(0,28));strip.paste(baseline,(width,28));strip.paste(improved,(width*2,28))
        draw=ImageDraw.Draw(strip);draw.text((8,7),'GT',fill='black');draw.text((width+8,7),'Legacy 6000',fill='black');draw.text((width*2+8,7),'Quality',fill='black');rows.append(strip)
    canvas=Image.new('RGB',(rows[0].width,sum(x.height for x in rows)),'white');y=0
    for row in rows:canvas.paste(row,(0,y));y+=row.height
    canvas.save(out/f'{label}_frozen_comparison.png')
panel('nerf',root/'outputs/quality/nerf_legacy6000/gaussian/final_test_renders',root/'outputs/quality/nerf_sh3_fixed/gaussian/final_test_renders',range(4))
panel('panorama',root/'outputs/quality/overlap_legacy6000/gaussian/final_test_renders',root/'outputs/quality/overlap_sh3_dense_pose/gaussian/final_test_renders',range(4))
old=Image.open(root/'outputs/kitchen/gaussian/train_preview.png').convert('RGB').resize((672,672),Image.Resampling.BICUBIC)
plain=Image.open(root/'outputs/quality/kitchen_highres/gaussian/train_preview.png').convert('RGB')
ba=Image.open(root/'outputs/quality/kitchen_highres_ba/gaussian/train_preview.png').convert('RGB')
strip=Image.new('RGB',(672*3,700),'white');draw=ImageDraw.Draw(strip)
for i,(name,image) in enumerate([('Legacy 336 (upsampled)',old),('Quality 672',plain),('Quality 672 + BA',ba)]):strip.paste(image,(i*672,28));draw.text((i*672+8,7),name,fill='black')
strip.save(out/'kitchen_training_comparison.png')
print(out)
