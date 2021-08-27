模型构建： UACL.py
数据集构建准备：dataset2txt.py
模型调参： UACL_train.py
模型训练： UACL_train.py


本程序提供MSTAR数据集作为默认数据，如需在其它数据集上使用请先运行按注释运行dataset2txt.py进行数据准备，然后在UACL_train.py中进行调参，在进行训练。

训练得到的预训练模型和微调后的分类模型默认都保存在checkpoints文件中


