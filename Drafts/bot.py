import twitchio
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Load the TfidfVectorizer
tfv_combined = joblib.load('/mnt/22A63810A637E2C9/Code/College/Twitch-Spam-detection/tfidf_vectorizer.joblib')

# Load the toxicity classification model
combined_label_model = joblib.load('/mnt/22A63810A637E2C9/Code/College/Twitch-Spam-detection/toxicity_classification_model.joblib')

# Twitch API credentials
client_id = ''
client_secret = ''
oauth_token = 'o'
initial_channels = ['']

# Create a Twitch client
bot = twitchio.Client(
    token=oauth_token,
    initial_channels=initial_channels
)

# Toxicity threshold
toxicity_threshold = 0.6

@bot.event
async def event_ready():
    print(f'Bot is connected and ready!')

@bot.event
async def event_message(ctx):
    # Access the message content
    message_content = ctx.content
    message_tfidf = tfv_combined.transform([message_content])
    toxicity_probability = combined_label_model.predict_proba(message_tfidf)[:, 1]

    # Print the received message content
    print(f'Received message from {ctx.author.name}: {message_content}')

    # Check toxicity and respond
    if toxicity_probability > toxicity_threshold:
        await ctx.channel.send(f'@{ctx.author.name}, your message is toxic. Please keep the chat clean!')

@bot.event
async def on_pubsub_message(data):
    # Check if the bot has joined the channel
    if data['type'] == 'USER_JOIN' and data['data']['user_name'].lower() == bot.nick.lower():
        print(f'Bot has joined the channel: {data["data"]["user_name"]}')

# Run the Twitch bot
print('Starting the bot...')
bot.run()
