import os
import random
from PIL import Image

data_path = ".\\data"
result_path = ".\\result"

if len(os.listdir(result_path)) == 0:
	
	for i in range(0,8):

		os.mkdir(os.path.join(result_path, str(i)))


for (subdirs, dirs, files) in os.walk(data_path):

	if subdirs == ".\\data":
		continue

	print(subdirs)

	for index, name in enumerate(os.listdir(subdirs)):

		img_name = os.path.join(subdirs,name[:len(name)-4])

		img = Image.open(os.path.join(subdirs,name))

		for i in range(1,101):
			
			rand_w = random.randrange(0,256-128)
			rand_h = random.randrange(0,256-128)

			img_crop = img.crop((rand_w,rand_h,rand_w+128,rand_h+128))

			img_name = img_name.replace("data","result")

			img_crop_name = img_name + "_{:03d}".format(int(i)) + ".tif"

			print(img_crop_name, img_crop.size)

			img_crop.save(img_crop_name)

