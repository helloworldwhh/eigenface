# coding:utf-8

from numpy import *
from numpy import linalg as la
from PIL import Image
import cv2
import os
import sys


def loadImageSet(add):  # 导入数据集
    FaceMat = mat(zeros((41, 112 * 92)))
    j = 0
    for i in os.listdir(add):
        img = cv2.imread(add + '\\' + i + '\\1.bmp', 0)
        FaceMat[j, :] = mat(img).flatten()
        j += 1
    return FaceMat


def ReconginitionVector(selecthr=0.8):
    # 第一步: 获得人脸数据集
    FaceMat = loadImageSet(sys.argv[2]).T
    # 第二步: 求均值人脸
    avgImg = mean(FaceMat, 1)
    # 第三步: 求偏差矩阵
    diffTrain = FaceMat - avgImg
    # 第四步: 求协方差矩阵的特征值和特征向量
    eigvals, eigVects = la.eig(mat(diffTrain.T * diffTrain))
    eigSortIndex = argsort(-eigvals)
    for i in range(shape(FaceMat)[1]):
        if (eigvals[eigSortIndex[:i]] / eigvals.sum()).sum() >= selecthr:
            eigSortIndex = eigSortIndex[:i]
            break
    covVects = diffTrain * eigVects[:, eigSortIndex]
    # avgImg 是均值图像，covVects是协方差矩阵的特征向量，diffTrain是偏差矩阵
    return avgImg, covVects, diffTrain


inpu = sys.argv[1]
count = 0
avgImg, FaceVector, diffTrain = ReconginitionVector(selecthr=float(inpu))
face = FaceVector.T[0]+FaceVector.T[1]+FaceVector.T[2]+FaceVector.T[3]+FaceVector.T[4]+FaceVector.T[5]+\
            FaceVector.T[6]+FaceVector.T[7]+FaceVector.T[8]+FaceVector.T[9]+FaceVector.T[10]
for i in range(FaceVector.shape[1]):
    aa = FaceVector.T[i].reshape(112, 92)
    cv2.imwrite(sys.argv[3] + '\\' + str(i) + '.bmp', aa)
tt = Image.fromarray(face.reshape(112, 92))
tt.show()

