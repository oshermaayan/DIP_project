#! /bin/sh
/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --net_arch skip --weight_init kaiming_uniform --input_resize_factor 2
/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --net_arch skip --weight_init kaiming_normal --input_resize_factor 2
/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --net_arch skip --weight_init kaiming_uniform --input_resize_factor 4
/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --net_arch skip --weight_init kaiming_normal --input_resize_factor 4
/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --net_arch skip --weight_init kaiming_uniform --input_noise_type n
/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --net_arch skip --weight_init kaiming_normal --input_noise_type n
/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --net_arch skip --weight_init kaiming_uniform --reg_noise_zero true
/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --net_arch skip --weight_init kaiming_normal --reg_noise_zero true
/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --net_arch skip --weight_init kaiming_uniform --reg_noise_large true
/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --net_arch skip --weight_init kaiming_normal --reg_noise_large true
/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --net_arch skip --weight_init kaiming_uniform --reg_noise_zero true --iter_num 4000
/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --net_arch skip --weight_init kaiming_normal --reg_noise_zero true --iter_num 4000
