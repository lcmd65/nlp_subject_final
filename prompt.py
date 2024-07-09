import json
import openai
import time

openai.api_key  = ""

def generate_chat_data(prompt, n_samples=1):
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates chat messages in the role of a loving couple."},
                    {"role": "user", "content": "Generate chat messages along with their emotional manipulation tags. Each message should be a tuple where the first element is the chat message string and the second element is a dictionary with keys 'normal' and 'manipulative', which are boolean values indicating whether the message is normal or manipulative. Example output:"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            return response.choices

def main():
    prompt = """
    ["Hey babe, I miss you so much! Can't wait to see you later 😘", { "cats": { "normal": true, "manipulative": false } }],
    ["Why didn't you reply to my text? Are you avoiding me?", { "cats": { "normal": false, "manipulative": true } }],
    """
    
    # Generate the data
    data = []
    
    for _ in range(100):  # Reduced to 100 iterations
        chat_data = generate_chat_data(prompt)
        for choice in chat_data:
            data.append(choice['message']['content'])

    with open("dataset.json", "w") as file:
        json.dump(data, file)

if __name__ == "__main__":
    main()
