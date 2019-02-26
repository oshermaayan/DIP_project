import numpy as np
import matplotlib.pyplot as plt
import pickle

def readPickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

baseDir = r'C:/DeepPriorProject/Final presentation/'
pathOrigSetts = baseDir+r'Original20KPsnr'
pathNoInputNoise = baseDir + r'noInputNoise'
pathWeightsNoise = baseDir + r'noisedWeights'
pathGradientNoise = baseDir + r'noisedGrads'
pathFeatureMapNoise = baseDir + r'noisedFeatureMaps'
pathInputSizeSmaller = baseDir + r'inputSize3'
pathDropGuard = baseDir + r'withDropGuard'
pathOrigSetts50K = baseDir + r'origSetts50K'
pathGradientClip50K = baseDir + r'gradientClip50K'

origPsnr = readPickle(pathOrigSetts)
noInputNoisePsnr = readPickle(pathNoInputNoise)
smallerInputDim = readPickle(pathInputSizeSmaller)
weightNoisePsnr = readPickle(pathWeightsNoise)
gradeNoisePsnr = readPickle(pathGradientNoise)
featureMapsPsnr = readPickle(pathFeatureMapNoise)
dropGuardPsnr = readPickle(pathDropGuard)
origPsnr50K = readPickle(pathOrigSetts50K)
gradClipPsnr = readPickle(pathGradientClip50K)

length = min(origPsnr50K.shape[0], gradClipPsnr.shape[0])
x_axis = [i for i in range(length)]

plt.plot(x_axis,origPsnr50K[0:length])
plt.plot(x_axis,gradClipPsnr[0:length])
plt.title("PSNR w.r.t iterations")
plt.xlabel("# Iterartions")
plt.ylabel("PSNR (dB)")
plt.legend(["Original settings","Normal init + gradient clipping"])
plt.savefig(baseDir + "gradClip.jpg")



print("Done")
