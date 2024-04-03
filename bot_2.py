from twitchio.ext import commands
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Load the TfidfVectorizer
tfv_combined = joblib.load('tfidf_vectorizer.joblib')

# Load the toxicity classification model
combined_label_model = joblib.load('toxicity_classification_model.joblib')

class ToxicityCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.Cog.event()
    async def event_message(self, message):
        # Access the message content
        message_content = message.content
        message_tfidf = tfv_combined.transform([message_content])
        toxicity_probability = combined_label_model.predict_proba(message_tfidf)[:, 1]
        print(f'Received message from {message.author.name}: {message_content}')

        # Check toxicity and respond
        toxicity_threshold = 0.65
        if toxicity_probability > toxicity_threshold:
            await message.channel.send(f'@{message.author.name}, your message is toxic. Please keep the chat clean!')

class Bot(commands.Bot):
    def __init__(self, token, prefix, initial_channels):
        super().__init__(token=token, prefix=prefix, initial_channels=initial_channels)
        self.add_cog(ToxicityCog(self)) 

    async def event_ready(self):
        # We are logged in and ready to chat and use commands...
        print(f'Logged in as | {self.nick}')
        print(f'User id is | {self.user_id}')
        print("Bot is Connected")

    @commands.command()
    async def hello(self, ctx: commands.Context):
        await ctx.send(f'Hello {ctx.author.name}!')

# Provide your access token, prefix, and initial channels
access_token = 'yourtoken here'
prefix = '?'
initial_channels = ['channel here']

# Create an instance of the Bot
bot = Bot(token=access_token, prefix=prefix, initial_channels=initial_channels)

# Run the bot
print("Bot is staring ...")
bot.run()
