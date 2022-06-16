from transformers import BertModel, BertTokenizerFast, BertForSequenceClassification
from GP_module import GlobalPointer
import json
import torch
import numpy as np
import pdb
from tqdm import  tqdm

MODEL_NAME = "../RoBERTa_zh_Large_PyTorch"
CBLUE_ROOT = "../data/CBLUEDatasets/CMeEE/CMeEE_test.json"

bert_model_path = MODEL_NAME #your RoBert_large path
save_model_path = '/dssg/home/acct-stu/stu928/zsy/CMEEE/ckpts/bert_GP_2022/checkpoint-12000/pytorch_model.bin'
device = torch.device("cuda:0")

LABEL  = ['dep', 'equ', 'mic', 'ite', 'dru', 'pro', 'sym', 'dis', 'bod'] 
EE_id2label  = [ L for L in LABEL]
EE_label2id  = {b: a for a, b in enumerate(EE_id2label)}

max_len = 100
ent2id, id2ent = EE_label2id, {}
for k, v in ent2id.items(): id2ent[v] = k

tokenizer = BertTokenizerFast.from_pretrained(bert_model_path)
encoder = BertModel.from_pretrained(bert_model_path)
model = GlobalPointer(encoder, 9 , 64).to(device)
model.load_state_dict(torch.load(save_model_path, map_location='cuda:0'))
model.eval()

def NER_RELATION(text, tokenizer, ner_model,  max_len=256):
    token2char_span_mapping = tokenizer(text, return_offsets_mapping=True, max_length=max_len)["offset_mapping"]
    new_span, entities= [], []
    for i in token2char_span_mapping:
        if i[0] == i[1]:
            new_span.append([])
        else:
            if i[0] + 1 == i[1]:
                new_span.append([i[0]])
            else:
                new_span.append([i[0], i[-1] - 1])

    encoder_txt = tokenizer.encode_plus(text, max_length=max_len)
    input_ids = torch.tensor(encoder_txt["input_ids"]).long().unsqueeze(0).cuda()
    token_type_ids = torch.tensor(encoder_txt["token_type_ids"]).unsqueeze(0).cuda()
    attention_mask = torch.tensor(encoder_txt["attention_mask"]).unsqueeze(0).cuda()
    print(input_ids.shape,token_type_ids.shape,attention_mask.shape)
    scores = ner_model.get_pred(input_ids, attention_mask, token_type_ids)[0].data.cpu().numpy()
    print(scores.shape)
    scores[:, [0, -1]] -= np.inf
    scores[:, :, [0, -1]] -= np.inf
    for l, start, end in zip(*np.where(scores > 0)):
        entities.append({"start_idx":new_span[start][0], "end_idx":new_span[end][-1], "type":id2ent[l]})
    return {"text":text, "entities":entities}

if __name__ == '__main__':
    all_ = []
    for d in tqdm(json.load(open(CBLUE_ROOT))):
        all_.append(NER_RELATION(d["text"], tokenizer= tokenizer, ner_model=model))
    json.dump(
        all_,
        open('CMeEE_test.json', 'w'),
        indent=4,
        ensure_ascii=False
    )