from bertviz import head_view, model_view
from transformers import BertTokenizer, BertModel, BertConfig, BertForSequenceClassification
import torch

model_file = './models/test_wnli/pytorch_model.bin'
config_file = './models/test_wnli/config.json'
vocab_file = './models/test_wnli/vocab.txt'
model_version = 'bert-base-uncased'
config = BertConfig.from_json_file(config_file)
#model = BertForSequenceClassification(config)
model = BertModel.from_pretrained(model_version,output_attentions=True)

state_dict = torch.load(model_file)
new_sd = dict()
for key in state_dict.keys():
    if key[0] == 'c':
        continue
    new_sd[key[5:]] = state_dict[key]
#print(new_sd.keys())

model.load_state_dict(new_sd)
tokenizer = BertTokenizer(vocab_file)
sentence_a = "The cat sat on the mat"
sentence_b = "The cat lay on the rug"
inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt')



input_ids = inputs['input_ids']
token_type_ids = inputs['token_type_ids']
print(input_ids.size())
print(input_ids)
tfidf_ids = torch.Tensor([[0,0,2,1,0,0,1,0,0,2,1,0,0,1,0]]).type(torch.int32)
print(tfidf_ids.size())
print(tfidf_ids)
attention = model(input_ids, token_type_ids=token_type_ids, tfidf_ids = tfidf_ids)[-1]
#print(attention)
sentence_b_start = token_type_ids[0].tolist().index(1)
input_id_list = input_ids[0].tolist() # Batch index 0
tokens = tokenizer.convert_ids_to_tokens(input_id_list) 

head_view(attention, tokens, sentence_b_start)


"""
model_version = 'bert-base-uncased'
model = BertModel.from_pretrained(model_version, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_version)
sentence_a = "The cat sat on the mat"
sentence_b = "The cat lay on the rug"
inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt')
input_ids = inputs['input_ids']
token_type_ids = inputs['token_type_ids']
attention = model(input_ids, token_type_ids=token_type_ids)[-1]
sentence_b_start = token_type_ids[0].tolist().index(1)
input_id_list = input_ids[0].tolist() # Batch index 0
tokens = tokenizer.convert_ids_to_tokens(input_id_list) 

head_view(attention, tokens, sentence_b_start)
"""
