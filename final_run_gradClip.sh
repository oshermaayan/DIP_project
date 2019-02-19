#!/bin/sh

# Comment-out less interesting tests
/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName gradClipNormalInit50K --weight_init normal --clipGradients True --iter_num 50000

#Different Weights inits
/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName OrigSetts50K --weight_init kaiming_uniform --iter_num 50000
