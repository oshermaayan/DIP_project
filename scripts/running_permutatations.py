#! usr/bin/python3.5

weight_init_types = ["uniform","normal","constant","xavier_uniform","xavier_normal","kaiming_uniform","kaiming_normal"]
#"dirac"
CNNs = ["skip","ResNet"]#,"UNet"]
images = ["data/sr/zebra_GT.png"]

print("#! /bin/sh")
code_base_path = r"/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py"
for image in images:
	for wit in weight_init_types:
		for CNN in CNNs:
			line = code_base_path + r" --file_path {image} --net_arch {net_arch} --weight_init {weight_init} --reg_noise_zero {regNoise}".format(image=image, net_arch=CNN, weight_init=wit, regNoise='True')
			print(line)
	

