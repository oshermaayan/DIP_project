#! /bin/sh
/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --net_arch skip --weight_init kaiming_uniform --input_noise_type n
/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --net_arch skip --weight_init kaiming_normal --input_noise_type n
