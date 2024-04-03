# %%
import numpy as np 
import pandas as pd 

# %% [markdown]
# ## Importing Necessary Libraries

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Text Processing Libraries

# %%
import re
import nltk
import string
import nlputils
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer

# %% [markdown]
# ## Data Visualization Libraries

# %%
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS

# %% [markdown]
# ## Machine Learning Libraries

# %%
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import (
    f1_score, precision_score, recall_score, precision_recall_curve,
    fbeta_score, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# %% [markdown]
# ## Miscellaneous Libraries

# %%
import os
import zipfile
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# ## Data Loading and Extraction

# %%
# Path to the Kaggle input directory
kaggle_input_path = '/mnt/22A63810A637E2C9/Code/College/Twitch-Spam-detection/Data/jigsaw-toxic-comment-classification-challenge'

# List files in the Kaggle input directory
files_in_directory = os.listdir(kaggle_input_path)

# Extract and load the data from the Kaggle zip files
for file_name in files_in_directory:
    if file_name.endswith('.zip'):
        zip_file_path = os.path.join(kaggle_input_path, file_name)
        output_dir = '/mnt/22A63810A637E2C9/Code/Twitch-Spam-detection/Data/output'
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
            extracted_files = zip_ref.namelist()
            for extracted_file in extracted_files:
                complete_path = os.path.join(output_dir, extracted_file)
                print("Extracted:", complete_path)

# Load the data into a Pandas DataFrame
data = pd.read_csv("/mnt/22A63810A637E2C9/Code/Twitch-Spam-detection/Data/output/train.csv")
data.sample(5)

# %% [markdown]
# ## Exploratory Data Analysis (EDA)

# %%
# Display basic information about the dataset
data.info()

# %%
# Check for missing values
data.isnull().sum()

# %%
# Display counts for each label category
label_counts = data.iloc[:, 2:].sum()

# %%
# Visualize label distribution
plt.figure(figsize=(8, 5))
sns.barplot(x=label_counts.index, y=label_counts.values, alpha=0.8)
plt.title("Labels per Classes")
plt.xlabel("Various Label Type")
plt.ylabel("Counts of the Labels")
plt.show()
df = pd.DataFrame(data)

# %% [markdown]
# ## Data Cleaning

# %%
# Sample comment before cleaning
df['comment_text'][10]

# %%
# Data cleaning functions
alphanumeric = lambda x: re.sub('\w*\d\w*', ' ', x)
punc_lower = lambda x: re.sub('[%s]' % re.escape(string.punctuation), ' ', x.lower())
remove_n = lambda x: re.sub("\n", " ", x)
remove_non_ascii = lambda x: re.sub(r'[^\x00-\x7f]', r' ', x)

# Apply data cleaning functions to the 'comment_text' column
df['comment_text'] = df['comment_text'].map(alphanumeric).map(punc_lower).map(remove_n).map(remove_non_ascii)

# %% [markdown]
# ## Class Balancing

# %%
Insulting_comment_df=df.loc[:,['id','comment_text','insult']]
Threatening_comment_df=df.loc[:,['id','comment_text','threat']]
IdentityHate_comment_df=df.loc[:,['id','comment_text','identity_hate']]
Obscene_comment_df=df.loc[:,['id','comment_text','obscene']]
Severetoxic_comment_df=df.loc[:,['id','comment_text','severe_toxic']]
Toxic_comment_df=df.loc[:,['id','comment_text','toxic']]

# %%
# Balancing the 'toxic' class
Toxic_comment_balanced_1 = Toxic_comment_df[Toxic_comment_df['toxic'] == 1].iloc[0:5000,:]
Toxic_comment_balanced_0 = Toxic_comment_df[Toxic_comment_df['toxic'] == 0].iloc[0:5000,:]
Toxic_comment_balanced = pd.concat([Toxic_comment_balanced_1,Toxic_comment_balanced_0])

# Balancing the 'severe_toxic' class
Severetoxic_comment_df_1 = Severetoxic_comment_df[Severetoxic_comment_df['severe_toxic'] == 1].iloc[0:1595,:]
Severetoxic_comment_df_0 = Severetoxic_comment_df[Severetoxic_comment_df['severe_toxic'] == 0].iloc[0:1595,:]
Severe_toxic_comment_balanced = pd.concat([Severetoxic_comment_df_1,Severetoxic_comment_df_0])

# Balancing the 'obscene' class
Obscene_comment_df_1 = Obscene_comment_df[Obscene_comment_df['obscene'] == 1].iloc[0:5000,:]
Obscene_comment_df_0 = Obscene_comment_df[Obscene_comment_df['obscene'] == 0].iloc[0:5000,:]
Obscene_comment_balanced = pd.concat([Obscene_comment_df_1,Obscene_comment_df_0])

# Balancing the 'threat' class
Threatening_comment_df_1 = Threatening_comment_df[Threatening_comment_df['threat'] == 1].iloc[0:478,:]
Threatening_comment_df_0 = Threatening_comment_df[Threatening_comment_df['threat'] == 0].iloc[0:478,:]
Threatening_comment_balanced = pd.concat([Threatening_comment_df_1,Threatening_comment_df_0])

# Balancing the 'insult' class
Insulting_comment_df_1 = Insulting_comment_df[Insulting_comment_df['insult'] == 1].iloc[0:5000,:]
Insulting_comment_df_0 = Insulting_comment_df[Insulting_comment_df['insult'] == 0].iloc[0:5000,:]
Insulting_comment_balanced = pd.concat([Insulting_comment_df_1,Insulting_comment_df_0])

# Balancing the 'identity_hate' class
IdentityHate_comment_df_1 = IdentityHate_comment_df[IdentityHate_comment_df['identity_hate'] == 1].iloc[0:1405,:]
IdentityHate_comment_df_0 = IdentityHate_comment_df[IdentityHate_comment_df['identity_hate'] == 0].iloc[0:5000,:]
IdentityHate_comment_balanced = pd.concat([IdentityHate_comment_df_1,IdentityHate_comment_df_0])

# %%
combined_df = pd.concat([Severe_toxic_comment_balanced, Threatening_comment_balanced, Insulting_comment_balanced, IdentityHate_comment_balanced, Toxic_comment_balanced], ignore_index=True)
combined_df['combined_label'] = combined_df.iloc[:, 2:].sum(axis=1)
combined_label_cv = cv_tf_train_test(combined_df, 'combined_label', TfidfVectorizer, (1, 1))
combined_label_cv.rename(columns={'F1 Score': 'F1 Score(combined_label)'}, inplace=True)


# %% [markdown]
# ## Word Cloud Visualization

# %%
# Word frequency analysis
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(" ".join(data['comment_text']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# %%
# Assuming you have a list of comment types
comment_types = ['severe_toxic', 'threat', 'insult', 'identity_hate', 'toxic']

# Set up subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
axes = axes.flatten()

for i, comment_type in enumerate(comment_types):
    # Select the DataFrame for the current comment type
    comment_df = combined_df[combined_df[comment_type] == 1]

    # Generate Word Cloud
    wordcloud = WordCloud(width=400, height=200, background_color='black').generate(" ".join(comment_df['comment_text']))

    # Plot the Word Cloud
    axes[i].imshow(wordcloud, interpolation='bilinear')
    axes[i].set_title(f'Word Cloud - {comment_type}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()


# %% [markdown]
# ## Text Vectorization and Model Training

# %%
def cv_tf_train_test(dataframe, label, vectorizer, ngram):
    # Split the data into X and y data sets
    X = dataframe.comment_text
    y = dataframe[label]

    # Split our data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

    # Using vectorizer and removing stopwords
    cv1 = vectorizer(ngram_range=(ngram), stop_words='english')

    # Transforming x-train and x-test
    X_train_cv1 = cv1.fit_transform(X_train)
    X_test_cv1  = cv1.transform(X_test)

    ## Machine learning models

    ## Logistic regression
    lr = LogisticRegression()
    lr.fit(X_train_cv1, y_train)

    ## k-nearest neighbours
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_cv1, y_train)

    ## Naive Bayes
    bnb = BernoulliNB()
    bnb.fit(X_train_cv1, y_train)

    ## Multinomial naive bayes
    mnb = MultinomialNB()
    mnb.fit(X_train_cv1, y_train)

    ## Support vector machine
    svm_model = LinearSVC()
    svm_model.fit(X_train_cv1, y_train)

    ## Random Forest
    randomforest = RandomForestClassifier(n_estimators=100, random_state=50)
    randomforest.fit(X_train_cv1, y_train)

    f1_score_data = {'F1 Score':[f1_score(lr.predict(X_test_cv1), y_test), f1_score(knn.predict(X_test_cv1), y_test),
                                f1_score(bnb.predict(X_test_cv1), y_test), f1_score(mnb.predict(X_test_cv1), y_test),
                                f1_score(svm_model.predict(X_test_cv1), y_test), f1_score(randomforest.predict(X_test_cv1), y_test)]}
    ## Saving f1 score results into a dataframe
    df_f1 = pd.DataFrame(f1_score_data, index=['Log Regression','KNN', 'BernoulliNB', 'MultinomialNB', 'SVM', 'Random Forest'])

    return df_f1

# %%
# Model training and evaluation for 'severe_toxic' class
severe_toxic_comment_cv = cv_tf_train_test(Severe_toxic_comment_balanced, 'severe_toxic', TfidfVectorizer, (1,1))
severe_toxic_comment_cv.rename(columns={'F1 Score': 'F1 Score(severe_toxic)'}, inplace=True)
severe_toxic_comment_cv

# %%
# Model training and evaluation for 'threat' class
threat_comment_cv = cv_tf_train_test(Threatening_comment_balanced, 'threat', TfidfVectorizer, (1,1))
threat_comment_cv.rename(columns={'F1 Score': 'F1 Score(threat)'}, inplace=True)
threat_comment_cv

# %%
# Model training and evaluation for 'insult' class
insult_comment_cv = cv_tf_train_test(Insulting_comment_balanced, 'insult', TfidfVectorizer, (1,1))
insult_comment_cv.rename(columns={'F1 Score': 'F1 Score(insult)'}, inplace=True)
insult_comment_cv

# %%
# Model training and evaluation for 'identity_hate' class
identity_hatecomment_cv = cv_tf_train_test(IdentityHate_comment_balanced, 'identity_hate', TfidfVectorizer, (1,1))
identity_hatecomment_cv.rename(columns={'F1 Score': 'F1 Score(identity_hate)'}, inplace=True)
identity_hatecomment_cv

# %%

X = Toxic_comment_balanced.comment_text
y = Toxic_comment_balanced['toxic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initiate a Tfidf vectorizer
tfv = TfidfVectorizer(ngram_range=(1,1), stop_words='english')

X_train_fit = tfv.fit_transform(X_train)
X_test_fit = tfv.transform(X_test)
randomforest = RandomForestClassifier(n_estimators=100, random_state=50)

randomforest.fit(X_train_fit, y_train)
randomforest.predict(X_test_fit)

# %%


# %%
print(combined_label_cv)


# %%
# Assuming you have the combined DataFrame and vectorizer already prepared
X_combined = combined_df.comment_text
y_combined = combined_df['combined_label']

X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(X_combined, y_combined, test_size=0.3, random_state=42)

# Initiate a Tfidf vectorizer
tfv_combined = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')

X_train_combined_fit = tfv_combined.fit_transform(X_train_combined)
X_test_combined_fit = tfv_combined.transform(X_test_combined)

# Initiate and train your model (Random Forest, for example)
combined_label_model = RandomForestClassifier(n_estimators=100, random_state=50)
combined_label_model.fit(X_train_combined_fit, y_train_combined)

# Evaluate the model
combined_label_predictions = combined_label_model.predict(X_test_combined_fit)
f1_combined_label = f1_score(combined_label_predictions, y_test_combined)

print(f'F1 Score for Combined Label: {f1_combined_label}')

new_comments = ['you should kill yourseld', 'have a great day']
new_comments_vect = tfv_combined.transform(new_comments)
combined_label_model.predict_proba(new_comments_vect)[:, 1]


# %%
new_comments = ['go kill yourself', 'nig']
new_comments_vect = tfv_combined.transform(new_comments)
combined_label_model.predict_proba(new_comments_vect)[:, 1]

# %%
comment1 = ['go kill yourself']
comment1_vect = tfv.transform(comment1)
randomforest.predict_proba(comment1_vect)[:,1]

# %%
comment2 = ['nig']
comment2_vect = tfv.transform(comment2)
randomforest.predict_proba(comment2_vect)[:,1]


