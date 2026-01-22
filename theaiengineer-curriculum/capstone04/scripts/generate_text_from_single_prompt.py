from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("OPENAI_API_KEY not set")
client = OpenAI(api_key = api_key)

response = client.responses.create(
    model="gpt-5-nano",
    input="write a one sentence bedtime store about a unicorn"
)

print(response.output_text);

