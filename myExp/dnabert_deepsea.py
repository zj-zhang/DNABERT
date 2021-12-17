#!/usr/bin/env python
# coding: utf-8

# # 1. Setup

# In[1]:


from transformers import BertConfig, BertForSequenceClassification, DNATokenizer
from transformers import glue_convert_examples_to_features as convert_examples_to_features
import torch
from tqdm import tqdm, trange


# # 2. Get DNABERT model

# In[2]:


# FZZ: get model from MODEL_CLASSES
config_class, model_class, tokenizer_class = BertConfig, BertForSequenceClassification, DNATokenizer
label_list = ['0','1']

config = config_class.from_pretrained(
    '/mnt/ceph/users/zzhang/DNABERT/myExp/6-new-12w-0',
    num_labels=len(label_list),
    finetuning_task='dnaprom',
    cache_dir=None,
)

config.hidden_dropout_prob = 0.1
config.attention_probs_dropout_prob = 0.1
config.split = int(100/512)
config.rnn = 'lstm'
config.num_rnn_layer = 2
config.rnn_dropout = 0
config.rnn_hidden = 768
config.output_hidden_states = True # add here FZZ

tokenizer = tokenizer_class.from_pretrained(
    'dna6',
    do_lower_case=False,
    cache_dir=None,
)
model = model_class.from_pretrained(
    '/mnt/ceph/users/zzhang/DNABERT/myExp/6-new-12w-0',
    from_tf=False,
    config=config,
    cache_dir=None,
)

_ = model.train()


# In[3]:


import h5py
import os

data_dir = "/mnt/ceph/users/zzhang/workspace_src/AMBER/examples/data/zero_shot_deepsea"
store = h5py.File(os.path.join(data_dir, "train.h5"), 'r')


# In[4]:


d = store['x'][101]
store.close()
# double-check DeepSEA letter index; should have a 100% match in hg19
''.join(['AGCT'[x] for x in d.argmax(axis=1)])


# In[5]:


from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures


class DeepSEA919Processor(DataProcessor):
    """Processor for the 2015 DeepSEA 919 multi-task data"""

    def get_labels(self):
        return ["0", "1"] 

    def get_train_examples(self, data_dir):
        print("LOOKING AT {}".format(os.path.join(data_dir, "train.h5")))
        return self._create_examples(self._read_h5(os.path.join(data_dir, "train.h5")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_h5(os.path.join(data_dir, "val.h5")), "dev")
    
    def _read_h5(self, fp):
        with h5py.File(fp, 'r') as store:
            return zip(*[[self._matrix_to_seq(x, letteridx='AGCT') for x in store['x'][()]], store['y'][()]])
    
    @staticmethod
    def _matrix_to_seq(d, letteridx='ACGT'):
        MAX_LEN = 512
        s = ''.join([letteridx[x] for x in d.argmax(axis=1)])
        slen = len(s)
        if slen > MAX_LEN:
            s = s[(slen//2-MAX_LEN//2) : (slen//2+MAX_LEN//2)]
        return s
            

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        TOKEN_SIZE = 6
        for (i, line) in tqdm(enumerate(lines)):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join([line[0][i:i+TOKEN_SIZE] for i in range(len(line[0])-TOKEN_SIZE)])
            label = str(line[1][0])
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


# In[6]:


processor = DeepSEA919Processor()
examples = processor.get_dev_examples(data_dir)
print(len(examples))


# In[7]:


max_length = 512
pad_on_left = False
pad_token = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
pad_token_segment_id = 0
output_mode = 'classification'

# from `InputExample` to `InputFeature`, add token_ids and attention masks
features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=max_length,
            output_mode=output_mode,
            pad_on_left=pad_on_left,  # pad on the left for xlnet
            pad_token=pad_token,
            pad_token_segment_id=pad_token_segment_id,)


# In[8]:


all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)


# In[9]:


dataset = torch.utils.data.TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)


# In[ ]:


data_iterator = tqdm(dataloader)

outputs = []
for batch in data_iterator:
    outputs.append([x.cpu().detach().numpy() for x in model.bert(input_ids=batch[0], attention_mask=batch[1])[2]])


# Annotation:
# - The layer number (13 layers)
# - The batch number (1 sentence)
# - The word / token number (100 tokens in our sentence)
# - The hidden unit / feature number (768 features)
# 

# In[ ]:


print(len(outputs))
print(outputs[0].shape)
print(outputs[1].shape)
print(len(outputs[2]))
print(outputs[2][0].shape)


# In[ ]:


outputs[2][9].cpu().detach().numpy()


# In[ ]:




