#!/bin/sh

#Different Weights inits
#/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName weightsInitKaimingUniform --weight_init kaiming_uniform --iter_num 25000 --psnrDropGuard True --saveWeights True

# Comment-out less interesting tests
#/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName weightsInitNormal --weight_init normal --iter_num 25000 --psnrDropGuard True --saveWeights True

#/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName weightsInitConstant --weight_init constant --iter_num 25000 --psnrDropGuard True --saveWeights True

/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName origSettsHighBetas5K --iter_num 20000


# Gradients noisening
/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName highBetas_NoisedGradient_std01_MaxGrad --noise_grad True --iter_num 15000 --noise_grad_std 0.1 --gradNoiseStdScale max --saveWeights True

/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName highBetas_NoisedGradient_std01AvgGrad --iter_num 25000 --noise_grad True --noise_grad_std 0.1 --gradNoiseStdScale mean --saveWeights True

/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName highBetas_NoisedGradient_std005_MaxGrad --noise_grad True --iter_num 20000 --noise_grad_std 0.05 --gradNoiseStdScale max --saveWeights True

/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName highBetas_NoisedGradient_std005AvgGrad --iter_num 20000 --noise_grad True --noise_grad_std 0.05 --gradNoiseStdScale mean --saveWeights True

# Weights noisening
/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName highBetas_NoisedWeights_std01_MaxWeight --noise_weights True --noise_weights_std 0.1 --iter_num 20000 --weightNoiseStdScale max --saveWeights True

/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName highBetas_NoisedWeights_std01_AvgWeight --noise_weights True --noise_weights_std 0.1 --iter_num 20000 --weightNoiseStdScale mean --saveWeights True

/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName highBetas_NoisedWeights_std005_MaxWeight --noise_weights True --noise_weights_std 0.05 --iter_num 20000 --weightNoiseStdScale max --saveWeights True

/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName highBetas_NoisedWeights_std005_AvgWeight --noise_weights True --noise_weights_std 0.05 --iter_num 20000 --weightNoiseStdScale mean --psnrDropGuard True --reg_noise_zero True

#Add noise to feature maps
/usr/bin/python3.5 /home/osherm/PycharmProjects/deep-image-prior/super-resolution.py --file_path data/sr/zebra_GT.png --simulationName highBetas_FeatureMapsAddedNoise --iter_num 20000 --addRegNoiseToFeatureMaps True --saveWeights True

