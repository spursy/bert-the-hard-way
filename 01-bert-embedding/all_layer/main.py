from transformers import BertModel, BertTokenizer
import torch

# After set output_hidden_stats=true, allow get all layer embedding
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sentence = 'I Love Pairs'

tokens = tokenizer.tokenize(sentence)

print(tokens)

# [CLS] tag at front, [SEP] tag at end
tokens = ['[CLS]'] + tokens + ['[SEP]']

print(tokens)

# if we need tag lenght is seven, add two [PAD] tag at the end 
tokens = tokens + ['[PAD]'] + ['[PAD]']

print(tokens)

# if tag is not [PAD], attention mask is 1
attention_mask = [1 if i != '[PAD]' else 0 for i in tokens]

print(attention_mask)

token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(token_ids)

token_ids = torch.tensor(token_ids).unsqueeze(0)

attention_mask = torch.tensor(attention_mask).unsqueeze(0)

hidden_rep, cls_head, hidden_states = model(token_ids, attention_mask = attention_mask, return_dict = False)

# torch.Size([1, 7, 768])
# [1, 7, 768] means [batch_size, sequence_length, hidden_size]
print(hidden_rep.shape)

# cls_head contain [CLS] tag embedding 
# [1, 768] means [batch_size, hidden_size]
print(cls_head.shape)

# cls head is sentence embedding
print(cls_head.shape)

print(len(hidden_states))

print(len(hidden_states[10].shape))
