
# coding: utf-8

# # Cleaning Your Data

# Let's take a web access log, and figure out the most-viewed pages on a website from it! Sounds easy, right?
# 
# Let's set up a regex that lets us parse an Apache access log line:

# In[1]:


import re

format_pat= re.compile(
    r"(?P<host>[\d\.]+)\s"
    r"(?P<identity>\S*)\s"
    r"(?P<user>\S*)\s"
    r"\[(?P<time>.*?)\]\s"
    r'"(?P<request>.*?)"\s'
    r"(?P<status>\d+)\s"
    r"(?P<bytes>\S*)\s"
    r'"(?P<referer>.*?)"\s'
    r'"(?P<user_agent>.*?)"\s*'
)


# Here's the path to the log file I'm analyzing:

# In[2]:


logPath = "access_log.txt"


# Now we'll whip up a little script to extract the URL in each access, and use a dictionary to count up the number of times each one appears. Then we'll sort it and print out the top 20 pages. What could go wrong?

# In[7]:


URLCounts = {}

with open(logPath, "r") as f:
    for line in (l.rstrip() for l in f):
        match= format_pat.match(line)
        if match:
            access = match.groupdict()
            request = access['request']
            (action, URL, protocol) = request.split()
            if URL in URLCounts:
                URLCounts[URL] = URLCounts[URL] + 1
            else:
                URLCounts[URL] = 1

results = sorted(URLCounts, key=lambda i: int(URLCounts[i]), reverse=True)

for result in results[:20]:
    print(result + ": " + str(URLCounts[result]))


# Hm. The 'request' part of the line is supposed to look something like this:
# 
# GET /blog/ HTTP/1.1
# 
# There should be an HTTP action, the URL, and the protocol. But it seems that's not always happening. Let's print out requests that don't contain three items:

# In[8]:


URLCounts = {}

with open(logPath, "r") as f:
    for line in (l.rstrip() for l in f):
        match= format_pat.match(line)
        if match:
            access = match.groupdict()
            request = access['request']
            fields = request.split()
            if (len(fields) != 3):
                print(fields)


# Huh. In addition to empty fields, there's one that just contains garbage. Well, let's modify our script to check for that case:

# In[9]:


URLCounts = {}

with open(logPath, "r") as f:
    for line in (l.rstrip() for l in f):
        match= format_pat.match(line)
        if match:
            access = match.groupdict()
            request = access['request']
            fields = request.split()
            if (len(fields) == 3):
                URL = fields[1]
                if URL in URLCounts:
                    URLCounts[URL] = URLCounts[URL] + 1
                else:
                    URLCounts[URL] = 1

results = sorted(URLCounts, key=lambda i: int(URLCounts[i]), reverse=True)

for result in results[:20]:
    print(result + ": " + str(URLCounts[result]))


# It worked! But, the results don't really make sense. What we really want is pages accessed by real humans looking for news from our little news site. What the heck is xmlrpc.php? A look at the log itself turns up a lot of entries like this:
# 
# 46.166.139.20 - - [05/Dec/2015:05:19:35 +0000] "POST /xmlrpc.php HTTP/1.0" 200 370 "-" "Mozilla/4.0 (compatible: MSIE 7.0; Windows NT 6.0)"
# 
# I'm not entirely sure what the script does, but it points out that we're not just processing GET actions. We don't want POSTS, so let's filter those out:

# In[10]:


URLCounts = {}

with open(logPath, "r") as f:
    for line in (l.rstrip() for l in f):
        match= format_pat.match(line)
        if match:
            access = match.groupdict()
            request = access['request']
            fields = request.split()
            if (len(fields) == 3):
                (action, URL, protocol) = fields
                if (action == 'GET'):
                    if URL in URLCounts:
                        URLCounts[URL] = URLCounts[URL] + 1
                    else:
                        URLCounts[URL] = 1

results = sorted(URLCounts, key=lambda i: int(URLCounts[i]), reverse=True)

for result in results[:20]:
    print(result + ": " + str(URLCounts[result]))


# That's starting to look better. But, this is a news site - are people really reading the little blog on it instead of news pages? That doesn't make sense. Let's look at a typical /blog/ entry in the log:
# 
# 54.165.199.171 - - [05/Dec/2015:09:32:05 +0000] "GET /blog/ HTTP/1.0" 200 31670 "-" "-"
# 
# Hm. Why is the user agent blank? Seems like some sort of malicious scraper or something. Let's figure out what user agents we are dealing with:

# In[11]:


UserAgents = {}

with open(logPath, "r") as f:
    for line in (l.rstrip() for l in f):
        match= format_pat.match(line)
        if match:
            access = match.groupdict()
            agent = access['user_agent']
            if agent in UserAgents:
                UserAgents[agent] = UserAgents[agent] + 1
            else:
                UserAgents[agent] = 1

results = sorted(UserAgents, key=lambda i: int(UserAgents[i]), reverse=True)

for result in results:
    print(result + ": " + str(UserAgents[result]))


# Yikes! In addition to '-', there are also a million different web robots accessing the site and polluting my data. Filtering out all of them is really hard, but getting rid of the ones significantly polluting my data in this case should be a matter of getting rid of '-', anything containing "bot" or "spider", and W3 Total Cache.

# In[12]:


URLCounts = {}

with open(logPath, "r") as f:
    for line in (l.rstrip() for l in f):
        match= format_pat.match(line)
        if match:
            access = match.groupdict()
            agent = access['user_agent']
            if (not('bot' in agent or 'spider' in agent or 
                    'Bot' in agent or 'Spider' in agent or
                    'W3 Total Cache' in agent or agent =='-')):
                request = access['request']
                fields = request.split()
                if (len(fields) == 3):
                    (action, URL, protocol) = fields
                    if (action == 'GET'):
                        if URL in URLCounts:
                            URLCounts[URL] = URLCounts[URL] + 1
                        else:
                            URLCounts[URL] = 1

results = sorted(URLCounts, key=lambda i: int(URLCounts[i]), reverse=True)

for result in results[:20]:
    print(result + ": " + str(URLCounts[result]))


# Now, our new problem is that we're getting a bunch of hits on things that aren't web pages. We're not interested in those, so let's filter out any URL that doesn't end in / (all of the pages on my site are accessed in that manner - again this is applying knowledge about my data to the analysis!)

# In[13]:


URLCounts = {}

with open(logPath, "r") as f:
    for line in (l.rstrip() for l in f):
        match= format_pat.match(line)
        if match:
            access = match.groupdict()
            agent = access['user_agent']
            if (not('bot' in agent or 'spider' in agent or 
                    'Bot' in agent or 'Spider' in agent or
                    'W3 Total Cache' in agent or agent =='-')):
                request = access['request']
                fields = request.split()
                if (len(fields) == 3):
                    (action, URL, protocol) = fields
                    if (URL.endswith("/")):
                        if (action == 'GET'):
                            if URL in URLCounts:
                                URLCounts[URL] = URLCounts[URL] + 1
                            else:
                                URLCounts[URL] = 1

results = sorted(URLCounts, key=lambda i: int(URLCounts[i]), reverse=True)

for result in results[:20]:
    print(result + ": " + str(URLCounts[result]))


# This is starting to look more believable! But if you were to dig even deeper, you'd find that the /feed/ pages are suspect, and some robots are still slipping through. However, it is accurate to say that Orlando news, world news, and comics are the most popular pages accessed by a real human on this day.
# 
# The moral of the story is - know your data! And always question and scrutinize your results before making decisions based on them. If your business makes a bad decision because you provided an analysis of bad source data, you could get into real trouble.
# 
# Be sure the decisions you make while cleaning your data are justifiable too - don't strip out data just because it doesn't support the results you want!

# ## Activity

# These results still aren't perfect; URL's that include "feed" aren't actually pages viewed by humans. Modify this code further to strip out URL's that include "/feed". Even better, extract some log entries for these pages and understand where these views are coming from.
