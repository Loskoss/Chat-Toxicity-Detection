## features

1. profanity Filter
2. no of message spam in 15 seconds
3. link spam filtering
4. ascii art filter
5. subscribe comment this kinda spam need to collect data set


~~~

Words/phrases
Acronyms (kys,kms etc)
False positives (words, places and names like 'mishit', 'scunthorpe' and 'titsworth')
URLs 
Personal information (email, address, phone etc - if applicable)

~~~

## Implementation

# **The main issue rn is Compilation of data and then marking it ham and spam EDA required**

### Profanity Filter
- Train the model on this list of profanity and get an score to delete or not delete the message 
- list can be found online ig

### no of messages
- every time a user joins the live stream add it viewer_mssg dict and everytime he texts add +1 the value of the userid dict
- Needs reset every 15 seconds
- VERY INEFFICIENT RN

### Link spam 
- can be trained on the same model with profanity

### Ascii art (Optional very tough)
- Need to scrap websites to get data on this and model training decisions need to be made , bayes classifier again but yes

### youtube spam comments
- dataset exits need to work on compiling the data into one file and then filter each into spam/ham for training
