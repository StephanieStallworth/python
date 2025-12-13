############# findall() example: no groups ###############
>>> import re

# Specify RegEx pattern 
>>> phoneRegex = re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')

# Search for all matches of pattern
>>> phoneRegex.search(resume)

>>> resume = '''
Stephanie C. Stallworth

Contact 1
937-475-5322
scstallworth1@gmail.com
Cincinnati, OH

Contact 2
555-555-5555
scstallworth1@gmail.com
Cincinnati, OH

SUMMARY 
MBA with 9+ years of quantitative experience in data-driven manufacturing and marketing tech industries. Expertise in consumer targeting and personalization with a proven ability to transform nuanced business challenges into scalable solutions. 

SKILLS
Languages: SQL, Bash, Python, R, Excel VBA 
Platforms: Apache (Airflow, Superset, Impala), Google Cloud Platform (BigQuery, AutoML, Looker) 
Data Visualization: Tableau, MicroStrategy, JDE Reports Now
Version Control: Git 	

EXPERIENCE                                            
Data Scientist, Data & Analytic Product (Jun 2019 - Present)					
Quotient Technology - Cincinnati, OH 						       
Responsible for supporting the development and optimization of Quotient's data product offerings.  Scope includes mining large data sets to build, test, and deploy models used to power personalized digital promotion and media targeting.  
Offer Recommendation Systems: Engineered Item-To-Item Collaborative Filtering Challenger algorithms to personalize offer placement in retailer galleries and surface the most relevant content for users. Key results include personalization algos outperforming non-personalized sort by 8%, representing an estimated 163K in average daily incremental coupon activations. 
Look-Alike Modeling: Automated execution of demographic look-alike models leveraged to achieve scale for media targeting campaigns - accelerating Quotient's audience generation capabilities to 12,000+ highly targetable media audiences deployed since launch.   
Propensity Scoring: Formulated propensity scoring algorithm used in decisioning for Quotient's Retail Ad Network to serve product ads co-branded with the network retailer a user is most likely to shop. Beta campaign resulted in $459K of total media attributable sales, fueling expansion of network to include retailers representing over $100B in grocery dollar sales to date.  

Data Analyst, Retail Analytics & Research (Jan 2018 - Jun 2019)				
Quotient Technology - Cincinnati, OH 						       
Delivered holistic insights to retail clients across drug, dollar, and grocery channels. Scope included digital promotion and media campaign measurement with deep dive analysis into attributable impact on consumer behavior and retail sales.  

Sales & Marketing Information Management Specialist (Jul 2014 - Dec 2017)                                          
Dayton Lamina Corporation -  Dayton, OH 
Headed sales analytics for tool manufacturer with $200 million in annual revenue and operating units in the U.S., Canada, and Mexico. Responsibilities included data extraction, cleaning, and reporting to inform strategic business decisions.  

EDUCATION
M.B.A, Wright State University - Dayton, OH  (Aug 2013)                                                                              
B.S. Finance, Wright State University - Dayton, OH  (Jun 2011)
'''

# Matches the first match, but not the later matches
# search() returns Match Objects 
>>> phoneRegex.search(resume)
# <re.Match object; span=(26, 38), match='937-475-5322'>

# Use findall() method to find all matches
# findall() returns a list of strings
>>> phoneRegex.findall(resume)
# ['937-475-5322', '555-555-5555']

############# findall() example: with groups ###############
# Create RegEx object with no groups
>>> phoneRegex = re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')

# If the Regular Expression pattern has zero or ONE group in them = list of strings
# Each string in that list will be the text that it found matching that pattern
>>> phoneRegex.findall(resume)
# ['937-475-5322', '555-555-5555']

# Create RegEx object has TWO OR MORE groups
>>> phoneRegex = re.compile(r'(\d\d\d)-(\d\d\d-\d\d\d\d)')

# If Regular Expressions pattern has two or more groups = list of tuples of strings
# Each tuple is a match inside the text
# String values in the tuples correspond to the groups inside of the RegEx object
>>> phoneRegex.findall(resume)
# [('937', '475-5322'), ('555', '555-5555')]

# If we wanted the entire string also, put it inside of its own group
phoneRegex = re.compile(r'((\d\d\d)-(\d\d\d-\d\d\d\d))')

# List of strings with 3 strings in each tuple, because now we have 3 groups
# Full number group, area code group, main number group
# Doesn't return a match object, returns a list of strings or a list of tuples of strings
>>> phoneRegex.findall(resume)
# [('937-475-5322', '937', '475-5322'), ('555-555-5555', '555', '555-5555')]