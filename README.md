# CMEEE
Knowledge Graph HM

## 预训练模型
我们在预训练模型部分分别尝试使用了BERT，RoBERTa，MedBERT和GigaWord，但是由于预存连模型过大，我们暂未上传至github，但其对应的下载链接如下：

    BERT：https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
    RoBERTa：https://pan.baidu.com/s/1MRDuVqUROMdSKr6HD9x1mw
    MedBERT：https://huggingface.co/trueto/medbert-base-chinese/tree/main
    GigaWord：https://pan.baidu.com/s/1pLO6T9D

## 模型部分
新使用到的模型为Global Point模型和FLAT模型，对应的模型结构在./src文件夹中

## 优化训练技巧
所使用到的对抗训练 ,随机参数平均和逐层学习率下降优化都是通过对Trainer进行重载后得到，新重载的Trainer在./src/NewTrainer.py文件中

## 训练模型
运行./src/run_cmeee.py文件即可

