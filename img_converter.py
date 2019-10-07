import glob
from PIL import Image

filename='soybean'
n=1
for img in glob.glob(filename+'/*.tif'):
    try:
        im = Image.open(img)
        im.save('./soybean_jpg/'+str(n)+'.jpg')
        n=n+1
    except Exception as e:
        print(e)