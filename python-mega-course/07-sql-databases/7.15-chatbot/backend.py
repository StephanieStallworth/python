import openai

class Chatbot:
    def __init__(self):
        openai.api_key = "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"

    # This method will be called from main.py
    def get_response(self, user_input):
        response = openai.Completion.create(
            engine="text-davinci-003", # model we'll use
            prompt=user_input,
            max_tokens=3000, # can set max of 4,080 tokens (words) for Bot to give longer answers
            temperature=0.5 # Range from 0-1, 0 produces more accurate answers but less diverse (more rigid)
        ).choices[0].text
        return response


if __name__ == "__main__":
    chatbot = ChatBot()
    response = chatbot.get_response("Write a joke about birds")

