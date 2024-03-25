import discord
import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Create a Discord client
client = discord.Client()

# Define a function to retrieve and pre-process the user's messages
async def get_user_messages(user_id):
    messages = []
    for channel in client.get_all_channels():
        async for message in channel.history():
            if message.author.id == user_id:
                # Pre-process the message text
                text = message.content.lower()
                text = re.sub(r'<.*?>', '', text)  # remove any HTML tags
                text = re.sub(r'http\S+', '', text)  # remove any links
                text = re.sub(r'[^\w\s]', '', text)  # remove any non-alphanumeric characters
                messages.append(text)
    # Concatenate all the pre-processed messages into a single document
    user_document = '\n'.join(messages)
    return user_document

# Define a function to fine-tune the GPT-2 model on the user's messages
def fine_tune_model(user_document):
    # Load the pre-trained GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Tokenize the user's document
    input_ids = tokenizer.encode(user_document, return_tensors='pt')

    # Fine-tune the model on the user's document
    model.train()
    model.zero_grad()
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    return model

# Listen for messages in the server
@client.event
async def on_message(message):
    # Check if the message mentions the user "ana"
    if 'ana' in message.content:
        user_document = await get_user_messages('903757836448837632')
        fine_tuned_model = fine_tune_model(user_document)
        # Generate a response using the fine-tuned GPT-2 model
        response = fine_tuned_model.generate(message.content)
        response_text = tokenizer.decode(response[0], skip_special_tokens=True)
        
        # Send the response back to the server
        await message.channel.send(response_text)

# Run the Discord client
client.run('MTA4MDkxODMwNDMyMDkyNTgzNg.Go31sM.yPwxP65Xtr63MLNNNeH5dRoTXI7yUOTQF860YM')
