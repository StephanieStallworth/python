
# coding: utf-8

# Instead of using fine-tuning, we'll use RAG to build our own "Commander Data" based on everything he ever said in the scripts.
# 
# To summarize the high level approach:
# 
# - We'll first parse all of the scripts to extract every line from Data, much as we did in the fine-tuning example.
# - Then we'll use the OpenAI embeddings API to compute embedding vectors for every one of his lines. This basically gives us similarity measures between every line.
# - RAG calls for use of a vector database to store these lines with the associated embedding vectors. To keep things simple, we'll use a local database called vectordb. There are plenty of cloud-based vector database services out there as well.
# - Then we'll make a little retrieval function that retrieves the N most-similar lines from the vector database for a given query
# - Those similar lines are then added as context to the prompt before it is handed off the the chat API.
# 
# I'm intentionally not using langchain or some other higher-level framework, because this is actually pretty simple without it.
# 
# First, let's install vectordb:

# In[1]:


get_ipython().system('pip install vectordb')


# Now we'll parse out all of the scripts and extract every line of dialog from "DATA". This is almost exactly the same code as from our fine tuning example's preprocessing script. Note you will need to upload all of the script files into a tng folder within your sample_data folder in your CoLab workspace first.
# 
# An archive can be found at https://www.st-minutiae.com/resources/scripts/ (look for "All TNG Epsiodes"), but you could easily adapt this to read scripts from your favorite character from your favorite TV show or movie instead.

# In[2]:


import os
import re
import random

dialogues = []

def strip_parentheses(s):
    return re.sub(r'\(.*?\)', '', s)

def is_single_word_all_caps(s):
    # First, we split the string into words
    words = s.split()

    # Check if the string contains only a single word
    if len(words) != 1:
        return False

    # Make sure it isn't a line number
    if bool(re.search(r'\d', words[0])):
        return False

    # Check if the single word is in all caps
    return words[0].isupper()

def extract_character_lines(file_path, character_name):
    lines = []
    with open(file_path, 'r') as script_file:
        try:
          lines = script_file.readlines()
        except UnicodeDecodeError:
          pass

    is_character_line = False
    current_line = ''
    current_character = ''
    for line in lines:
        strippedLine = line.strip()
        if (is_single_word_all_caps(strippedLine)):
            is_character_line = True
            current_character = strippedLine
        elif (line.strip() == '') and is_character_line:
            is_character_line = False
            dialog_line = strip_parentheses(current_line).strip()
            dialog_line = dialog_line.replace('"', "'")
            if (current_character == 'DATA' and len(dialog_line)>0):
                dialogues.append(dialog_line)
            current_line = ''
        elif is_character_line:
            current_line += line.strip() + ' '

def process_directory(directory_path, character_name):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):  # Ignore directories
            extract_character_lines(file_path, character_name)



# In[3]:


process_directory("./sample_data/tng", 'DATA')


# Let's do a little sanity check to make sure the lines imported correctly, and print out the first one.

# In[4]:


print (dialogues[0])


# Now we'll set up what a document in our vector database looks like; it's just a string and its embedding vector. We're going with 128 dimensions in our embeddings to keep it simple, but you could go larger. The OpenAI model we're using has up to 1536 I believe.

# In[5]:


from docarray import BaseDoc
from docarray.typing import NdArray

embedding_dimensions = 128

class DialogLine(BaseDoc):
  text: str = ''
  embedding: NdArray[embedding_dimensions]


# It's time to start computing embeddings for each line in OpenAI, so let's make sure OpenAI is installed:

# In[6]:


get_ipython().system('pip install openai --upgrade')


# Let's initialize the OpenAI client, and test creating an embedding for a single line of dialog just to make sure it works.
# 
# You will need to provide your own OpenAI secret key here. To use this code as-is, click on the little key icon in CoLab and add a "secret" for OPENAI_API_KEY that points to your secret key.

# In[7]:


from google.colab import userdata

from openai import OpenAI
client = OpenAI(api_key=userdata.get('OPENAI_API_KEY'))

embedding_model = "text-embedding-3-small"

response = client.embeddings.create(
    input=dialogues[1],
    dimensions=embedding_dimensions,
    model= embedding_model
)

print(response.data[0].embedding)


# Let's double check that we do in fact have embeddings of 128 dimensions as we specified.

# In[8]:


print(len(response.data[0].embedding))


# OK, now let's compute embeddings for every line Data ever said. The OpenAI API currently can't handle computing them all at once, so we're breaking it up into 128 lines at a time here.

# In[9]:


#Generate embeddings for everything Data ever said, 128 lines at a time.
embeddings = []

for i in range(0, len(dialogues), 128):
  dialog_slice = dialogues[i:i+128]
  slice_embeddings = client.embeddings.create(
    input=dialog_slice,
    dimensions=embedding_dimensions,
    model=embedding_model
  )

  embeddings.extend(slice_embeddings.data)


# Let's check how many embeddings we actually got back in total.

# In[10]:


print (len(embeddings))


# Now let's insert every line and its embedding vector into our vector database. The syntax here will vary depending on which vector database you are using, but it's pretty simple.
# 
# With vectordb, we need to set up a workspace folder for it to use, so be sure to create that within your CoLab drive first.

# In[11]:


from docarray import DocList
import numpy as np
from vectordb import InMemoryExactNNVectorDB

# Specify your workspace path
db = InMemoryExactNNVectorDB[DialogLine](workspace='./sample_data/workspace')

# Index our list of documents
doc_list = [DialogLine(text=dialogues[i], embedding=embeddings[i].embedding) for i in range(len(embeddings))]
db.index(inputs=DocList[DialogLine](doc_list))


# Let's try querying our vector database for lines similar to a query string.
# 
# First we need to compute the embedding vector for our query string, then we'll query the vector database for the top 10 matches based on the similarities encoded by their embedding vectors.

# In[12]:


# Perform a search query
queryText = 'Lal, my daughter'
response = client.embeddings.create(
    input=queryText,
    dimensions=embedding_dimensions,
    model=embedding_model
)
query = DialogLine(text=queryText, embedding=response.data[0].embedding)
results = db.search(inputs=DocList[DialogLine]([query]), limit=10)

# Print out the matches
for m in results[0].matches:
  print(m)


# Let's put it all together! We'll write a generate_response function that:
# 
# - Computes an embedding for the query passed in
# - Queries our vector database for the 10 most similar lines to that query (you could experiment with using more or less)
# - Constructs a prompt that adds in these similar lines as context, to try and nudge ChatGPT in the right direction using our external data
# - Feeds to augmented prompt into the chat completions API to get our response.
# 
# That's RAG!

# In[13]:


import openai

def generate_response(question):
    # Search for similar dialogues in the vector DB
    response = client.embeddings.create(
        input=question,
        dimensions=embedding_dimensions,
        model=embedding_model
    )

    query = DialogLine(text=queryText, embedding=response.data[0].embedding)
    results = db.search(inputs=DocList[DialogLine]([query]), limit=10)

    # Extract relevant context from search results
    context = "\n"
    for result in results[0].matches:
      context += "\"" + result.text + "\"\n"
#    context = '/n'.join([result.text for result in results[0].matches])

    prompt = f"Lt. Commander Data is asked: '{question}'. How might Data respond? Take into account Data's previous responses similar to this topic, listed here: {context}"
    print("PROMPT with RAG:\n")
    print(prompt)

    print("\nRESPONSE:\n")
    # Use OpenAI API to generate a response based on the context
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "You are Lt. Cmdr. Data from Star Trek: The Next Generation."},
        {"role": "user", "content": prompt}
      ]
    )

    return (completion.choices[0].message.content)




# Let's try it out! Note that the final response does seem to be drawing from the model's own training, but it is building upon the specific lines we gave it, allowing us to have some control over its output.

# In[14]:


print(generate_response("Tell me about your daughter, Lal."))

