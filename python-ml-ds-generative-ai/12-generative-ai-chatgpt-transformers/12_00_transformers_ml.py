
# coding: utf-8

# In[ ]:


get_ipython().system('pip install git+https://github.com/huggingface/transformers')


# In[ ]:


get_ipython().system('pip install jupyterlab ipywidgets bertviz xformers evaluate matplotlib')


# # Tokenizers

# In[ ]:


from transformers import BertModel, BertTokenizer

modelName = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(modelName)
model = BertModel.from_pretrained(modelName)


# In[ ]:


tokenized = tokenizer("I read a good novel.")
print(tokenized)


# In[ ]:


tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"])
print(tokens)


# # Positional Encoding

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

def encodePositions(num_tokens, depth, n=10000):
    positionalMatrix = np.zeros((num_tokens, depth))
    for row in range(num_tokens):
        for col in np.arange(int(depth/2)):
            denominator = np.power(n, 2*col/depth)
            positionalMatrix[row, 2*col] = np.sin(row/denominator)
            positionalMatrix[row, 2*col+1] = np.cos(row/denominator)
    return positionalMatrix


# In[ ]:


positionalMatrix = encodePositions(50, 256)
fig = plt.matshow(positionalMatrix)
plt.gcf().colorbar(fig)


# # Self-Attention

# In[ ]:


from bertviz.transformers_neuron_view import BertModel, BertTokenizer
from bertviz.neuron_view import show

tokenizer_viz = BertTokenizer.from_pretrained(modelName)
model_viz = BertModel.from_pretrained(modelName)
show(model_viz, "bert", tokenizer_viz, "I read a good novel.", display_mode="light", head=11)


# In[ ]:


show(model_viz, "bert", tokenizer_viz, "Attention is a novel idea.", display_mode="light", head=11)


# Also play with https://huggingface.co/spaces/exbert-project/exbert

# # GPT2 model (137M parameters)

# In[ ]:


from transformers import pipeline
generator = pipeline('text-generation', model='gpt2')
generator("I read a good novel.", max_length=30, num_return_sequences=5)


# In[ ]:


generator("This movie seemed really long.", max_length=300, num_return_sequences=5)


# In[ ]:


generator("Star Trek" , max_length=100, num_return_sequences=5)


# # GPT2-Large model (812M parameters)

# In[ ]:


generator = pipeline('text-generation', model='gpt2-large')
generator("I read a good novel.", max_length=30, num_return_sequences=5)


# ## Fine-Tuning GPT2

# In[ ]:


get_ipython().system('wget https://raw.githubusercontent.com/huggingface/transformers/main/examples/pytorch/language-modeling/run_clm.py')


# In[ ]:


get_ipython().system('pip install transformers[torch]')


# In[ ]:


get_ipython().run_cell_magic('bash', '', '\npython run_clm.py \\\n    --model_name_or_path gpt2 \\\n    --dataset_name imdb \\\n    --per_device_train_batch_size 8 \\\n    --per_device_eval_batch_size 8 \\\n    --do_train \\\n    --do_eval \\\n    --output_dir /tmp/test-clm')


# In[ ]:


from transformers import pipeline
from transformers import GPT2Tokenizer, GPT2LMHeadModel

dir = "/tmp/test-clm"
generator = pipeline('text-generation', model=GPT2LMHeadModel.from_pretrained(dir), tokenizer=GPT2Tokenizer.from_pretrained(dir))
generator("I read a good novel.", max_length=30, num_return_sequences=5)


# In[ ]:


generator("This movie seemed really long.", max_length=300, num_return_sequences=5)


# In[ ]:


generator("Star Trek", max_length=100, num_return_sequences=5)

