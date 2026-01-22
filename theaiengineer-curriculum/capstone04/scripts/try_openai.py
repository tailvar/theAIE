from openai import OpenAI

client=OpenAI()

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain gamma scalping in one paragraph."}
    ],
)

print(response.choices[0].message.content)