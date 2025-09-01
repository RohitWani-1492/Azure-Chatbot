from flask import Flask, render_template, request, jsonify
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os


load_dotenv()

app = Flask(__name__)


chat_model = AzureChatOpenAI(
    azure_deployment="gpt-4o",  
    model="gpt-4o",
    temperature=0.7,
    top_p=1.0,
    max_tokens=2000,
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("OPENAI_API_VERSION")
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')

    try:
        response = chat_model.invoke(user_input)
        return jsonify({'response': response.content})
    except Exception as e:
        return jsonify({'response': f"Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
