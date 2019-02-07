from numpy import *
from PIL import Image
import cv2
import os
import sys

# 文件读入序号纠正
num = [1,10,11,12,13,14,15,16,17,18,19,2,20,21,22,23,24,25,26,27,28,29,3,30,31,32,33,34,35,36,37,38,39,4,40,41,5,6,7,8,9]


def loadImageSet(add):  # 导入数据集
    FaceMat = mat(zeros((41, 112 * 92)))
    j = 0
    for i in os.listdir(add):
        img = cv2.imread(add + '\\' + i + '\\1.bmp', 0)
        FaceMat[j, :] = mat(img).flatten()
        j += 1
    return FaceMat


def judgeFace(judgeImg, FaceVector, avgImg, diffTrain):  # 识别人脸
    diff = judgeImg.T - avgImg
    weiVec = FaceVector.T * diff  # 计算特征脸表示的权重向量
    res = 0
    resVal = inf
    # 找到欧式距离最小的特征脸
    for i in range(41):
        TrainVec = FaceVector.T * diffTrain[:, i]
        if (array(weiVec - TrainVec) ** 2).sum() < resVal:
            res = i
            resVal = (array(weiVec - TrainVec) ** 2).sum()
    return num[res], res, weiVec


inpu = 'att_faces\\s'+str(sys.argv[1])+'\\'+str(sys.argv[2])+'.bmp'
FaceMat = loadImageSet("E:\\Desktop\\wh\\python project\\eigenface\\att_faces").T
avgImg = mean(FaceMat, 1)
diffTrain = FaceMat - avgImg
count = 0
loadname = "E:\\Desktop\\wh\\python project\\eigenface\\"
judgeimg = cv2.imread(loadname+inpu, 0)  # 待识别图像
judgeimg = mat(judgeimg).flatten()
diff = judgeimg-avgImg
le = len(os.listdir(loadname+'model\\'))
FaceVector = mat(zeros((le, 112 * 92)))  # 特征脸矩阵
j = 0
for i in range(0, le):
    img = cv2.imread(loadname+'model\\'+str(i)+'.bmp', 0)
    FaceVector[j, :] = mat(img).flatten()
    j += 1
res, num, wev = judgeFace(judgeimg, FaceVector.T, avgImg, diffTrain)
tt = wev.T*FaceVector  # 识别结果叠加
tt = Image.fromarray(tt.reshape(112, 92))
tt.show()
rr = Image.fromarray(FaceMat.T[num].reshape(112, 92))  # 对应人脸库中最相似的脸
rr.show()
print(res)