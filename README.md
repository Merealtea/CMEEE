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
使用到的对抗训练代码实现在./src/adversarial.py中，随机参数平均和逐层学习率下降优化都是通过对Trainer进行重载后得到，新重载的Trainer在./src/NewTrainer.py文件中

## 训练模型
运行./src/run_cmeee.py文件即可

## Requirements
    
    datasets==1.10.0
    FastNLP==0.7.0
    gensim==4.1.2
    huggingface-hub==0.0.19
    jieba==0.42.1
    nltk==3.7
    pytorch-crf==0.7.2
    tensorflow-estimator==1.13.0
    tokenizers==0.10.3
    torch==1.11.0
    torchaudio==0.11.0
    torchvision==0.12.0
    tqdm==4.62.3
    transformers==4.11.0
