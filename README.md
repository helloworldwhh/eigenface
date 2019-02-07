# eigenface
实现eigenface人脸识别的训练与识别过程

运行mytrain.py:
python  mytrain.py  0.9  att_faces  model
#其中0.9为阈值，可选
#att_faces为人脸库，包含41个人脸，每个10张，其中第一张用来做训练
#model为输出训练结果特征脸的文件夹

运行mytest.py:
python mytest.py 41 2
#其中41为人脸集的序号，可从1-41中选取
#2为该人脸集中脸的照片的序号，可从2-10中选取
