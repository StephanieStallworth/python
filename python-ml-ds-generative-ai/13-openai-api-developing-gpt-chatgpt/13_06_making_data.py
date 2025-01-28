
# coding: utf-8

# Install the latest OpenAI package...

# In[1]:


get_ipython().system('pip install openai --upgrade')


# In[2]:


import os
from openai import OpenAI
client = OpenAI()


# Upload our training and evaluation files, in chat completion format:

# In[8]:


client.files.create(
  file=open("./DATA_train.jsonl", "rb"),
  purpose='fine-tune'
)


# In[9]:


client.files.create(
  file=open("./DATA_eval.jsonl", "rb"),
  purpose='fine-tune'
)


# Check the status of these files by copying in the returned ID's above. If there are JSON errors they will be reported here.

# In[11]:


client.files.retrieve("file-UqPVnkk9z8Q74BEUqPlnhjHL")


# Start our fine tuning job! Copy in the ID's for our uploaded training and validation files.

# In[12]:


client.fine_tuning.jobs.create(training_file="file-9lI2ovFA1UJskgOPpxDTwEhG", validation_file="file-UqPVnkk9z8Q74BEUqPlnhjHL", model="gpt-3.5-turbo")


# Get general info about this job.

# In[13]:


client.fine_tuning.jobs.retrieve("ftjob-mQlhbPB5vsog1SeDLNx2xAMj")


# Monitor its progress. When done, you can use the resulting fine tuned model ID in the playground (or the API)

# In[32]:


client.fine_tuning.jobs.list_events(id="ftjob-mQlhbPB5vsog1SeDLNx2xAMj", limit=10)


# For comparison, see how the non-fine-tuned GPT model does:

# In[30]:


completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "Data is an android in the TV series Star Trek: The Next Generation."},
    {"role": "user", "content": "PICARD: Mr. Data, scan for lifeforms."}
  ]
)

print(completion.choices[0].message)


# When it's done, try our fine-tuned model! Copy in our fine tuned ID.

# In[31]:


completion = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-0613:sundog-software-llc::7qiBf2gI",
  messages=[
    {"role": "system", "content": "Data is an android in the TV series Star Trek: The Next Generation."},
    {"role": "user", "content": "PICARD: Mr. Data, scan for lifeforms."}
  ]
)

print(completion.choices[0].message)

