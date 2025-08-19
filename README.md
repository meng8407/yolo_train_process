# yolo_train_process

### 这是一个yolo 一键训练的代码，指定文件和路径即可实现一键训练，
##### 主文件是train.py文件，
1.替换input_dir地址为一个包含xml文件和图片地址的文件路径，
2.output_dir为生成yolo训练集的路径，
3.替换class_name为自己的标签名称，可以是多标签，
4.fs_trans，split_five_equal 参数是选择是否使用放射变换或者五等份切分对训练数据集进行增强
5.指定训练中的batch_size ,imgsz，和要训练的yolo模型。

python count_classes.py 查看文件夹中类别标签，获取class_name


#### 更换路径后直接执行，python train.py  

