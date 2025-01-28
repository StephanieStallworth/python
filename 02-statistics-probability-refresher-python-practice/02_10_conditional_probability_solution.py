
# coding: utf-8

# # Conditional Probability Solution

# First we'll modify the code to have some fixed purchase probability regardless of age, say 40%:

# In[1]:


from numpy import random
random.seed(0)

totals = {20:0, 30:0, 40:0, 50:0, 60:0, 70:0}
purchases = {20:0, 30:0, 40:0, 50:0, 60:0, 70:0}
totalPurchases = 0
for _ in range(100000):
    ageDecade = random.choice([20, 30, 40, 50, 60, 70])
    purchaseProbability = 0.4
    totals[ageDecade] += 1
    if (random.random() < purchaseProbability):
        totalPurchases += 1
        purchases[ageDecade] += 1


# Next we will compute P(E|F) for some age group, let's pick 30 year olds again:

# In[2]:


PEF = float(purchases[30]) / float(totals[30])
print("P(purchase | 30s): " + str(PEF))


# Now we'll compute P(E)

# In[3]:


PE = float(totalPurchases) / 100000.0
print("P(Purchase):" + str(PE))


# P(E|F) is pretty darn close to P(E), so we can say that E and F are likely indepedent variables.
