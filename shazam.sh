#!/bin/sh

/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName weightsInitKaimingUniform --weight_init kaiming_uniform

/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName weightsInitKaimingNormal --weight_init kaiming_normal

/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName weightsInitNormal --weight_init normal

/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName weightsInitConstant --weight_init constant

/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName weightsInitXavierUniform --weight_init xavier_uniform

/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName weightsInitXavierNormal --weight_init xavier_normal

/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName NoisedGradient_std01 --noise_grad True --noise_grad_std 0.1

/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName NoisedGradient_std001 --noise_grad True --noise_grad_std 0.01

/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName NoisedGradient_std0001 --noise_grad True --noise_grad_std 0.001

/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName NoisedLr_std01 --noise_lr True --noise_grad_std 0.1

/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName NoisedLr_std001 --noise_lr True --noise_grad_std 0.01

/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName NoisedLr_std0001 --noise_lr True --noise_grad_std 0.001

/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName NoisedWeights_std01 --noise_weights True --noise_weights_std 0.1

/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName NoisedWeights_std001 --noise_weights True --noise_weights_std 0.01

/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName NoisedWeights_std0001 --noise_weights True --noise_weights_std 0.001

