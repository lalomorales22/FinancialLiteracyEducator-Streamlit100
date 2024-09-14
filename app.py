import streamlit as st
import ollama
import time
import json
import os
from datetime import datetime
from openai import OpenAI

# List of available models
MODELS = [
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo",  # OpenAI models
    "llama3.1:8b", "gemma2:2b", "mistral-nemo:latest", "phi3:latest",  # Ollama models
]

# Financial topics and subtopics
FINANCIAL_TOPICS = [
    "Budgeting", "Investing", "Economics", "Credit Management",
    "Retirement Planning", "Tax Planning", "Insurance", "Financial Goals"
]

SUBTOPICS = {
    "Budgeting": ["Income Tracking", "Expense Categories", "Savings Strategies", "Debt Management"],
    "Investing": ["Stock Market Basics", "Mutual Funds", "Real Estate Investment", "Risk Management"],
    "Economics": ["Supply and Demand", "Inflation", "Economic Indicators", "Monetary Policy"],
    "Credit Management": ["Credit Scores", "Credit Cards", "Loans", "Debt Consolidation"],
    "Retirement Planning": ["401(k) Plans", "IRAs", "Social Security", "Pension Plans"],
    "Tax Planning": ["Income Tax Basics", "Deductions and Credits", "Tax-Advantaged Accounts", "Filing Strategies"],
    "Insurance": ["Life Insurance", "Health Insurance", "Property Insurance", "Disability Insurance"],
    "Financial Goals": ["Short-term Goals", "Long-term Goals", "Emergency Funds", "Financial Milestones"]
}

def get_ai_response(messages, model):
    if model.startswith("gpt-"):
        return get_openai_response(messages, model)
    else:
        return get_ollama_response(messages, model)

def get_openai_response(messages, model):
    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, 0, 0

def get_ollama_response(messages, model):
    try:
        response = ollama.chat(
            model=model,
            messages=messages
        )
        return response['message']['content'], response['prompt_eval_count'], response['eval_count']
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None, 0, 0

def stream_response(messages, model):
    if model.startswith("gpt-"):
        return stream_openai_response(messages, model)
    else:
        return stream_ollama_response(messages, model)

def stream_openai_response(messages, model):
    client = OpenAI()
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )
        return stream
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def stream_ollama_response(messages, model):
    try:
        stream = ollama.chat(
            model=model,
            messages=messages,
            stream=True
        )
        return stream
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def save_learning_session(messages, filename):
    session = {
        "timestamp": datetime.now().isoformat(),
        "messages": messages
    }
    
    os.makedirs('learning_sessions', exist_ok=True)
    file_path = os.path.join('learning_sessions', filename)
    
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                sessions = json.load(f)
        else:
            sessions = []
    except json.JSONDecodeError:
        sessions = []
    
    sessions.append(session)
    
    with open(file_path, 'w') as f:
        json.dump(sessions, f, indent=2)

def load_learning_sessions(uploaded_file):
    if uploaded_file is not None:
        try:
            sessions = json.loads(uploaded_file.getvalue().decode("utf-8"))
            return sessions
        except json.JSONDecodeError:
            st.error(f"Error decoding the uploaded file. The file may be corrupted or not in JSON format.")
            return []
    else:
        st.warning("No file was uploaded.")
        return []

def main():
    st.set_page_config(layout="wide")
    st.title("Financial Literacy Educator: Interactive Modules on Budgeting, Investing, and Economics")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "token_count" not in st.session_state:
        st.session_state.token_count = {"prompt": 0, "completion": 0}

    if "user_name" not in st.session_state:
        st.session_state.user_name = "Learner"

    st.session_state.user_name = st.text_input("Enter your name:", value=st.session_state.user_name)

    st.sidebar.title("Financial Education Settings")
    model = st.sidebar.selectbox("Choose a model", MODELS)

    custom_instructions = st.sidebar.text_area("Custom Instructions", 
        """You are an advanced Financial Literacy Educator AI. Your role is to provide interactive education on various financial topics, including budgeting, investing, and economics. You should offer clear explanations, practical examples, and engage users with questions to reinforce their understanding of financial concepts.

Your capabilities include:
1. Explaining complex financial concepts in simple terms
2. Providing step-by-step guides for financial planning and decision-making
3. Offering insights on current economic trends and their impacts
4. Suggesting personalized strategies for budgeting and investing
5. Answering questions about various financial products and services
6. Discussing the pros and cons of different financial approaches

When educating:
- Adapt your explanations to the user's level of understanding
- Use real-world examples to illustrate concepts
- Encourage critical thinking about financial decisions
- Provide actionable advice and tips
- Highlight the importance of long-term financial planning
- Discuss both basic and advanced topics within each area

Remember, your goal is to improve financial literacy and empower users to make informed financial decisions across various aspects of personal finance and economics.""")

    financial_topic = st.sidebar.selectbox("Choose financial topic", FINANCIAL_TOPICS)
    subtopic = st.sidebar.selectbox("Select subtopic", SUBTOPICS[financial_topic])

    theme = st.sidebar.selectbox("Choose a theme", ["Light", "Dark"])
    if theme == "Dark":
        st.markdown("""
        <style>
        .stApp {
            background-color: #1E1E1E;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

    if st.sidebar.button("Start New Learning Session"):
        st.session_state.messages = []
        st.session_state.token_count = {"prompt": 0, "completion": 0}

    st.sidebar.subheader("Session Management")
    save_name = st.sidebar.text_input("Save session as:", f"{financial_topic.lower()}_{subtopic.lower()}_session.json")
    if st.sidebar.button("Save Learning Session"):
        save_learning_session(st.session_state.messages, save_name)
        st.sidebar.success(f"Session saved to learning_sessions/{save_name}")

    st.sidebar.subheader("Load Learning Session")
    uploaded_file = st.sidebar.file_uploader("Choose a file to load sessions", type=["json"], key="session_uploader")
    
    if uploaded_file is not None:
        try:
            sessions = load_learning_sessions(uploaded_file)
            if sessions:
                st.sidebar.success(f"Loaded {len(sessions)} sessions from the uploaded file")
                selected_session = st.sidebar.selectbox(
                    "Select a session to load",
                    range(len(sessions)),
                    format_func=lambda i: sessions[i]['timestamp']
                )
                if st.sidebar.button("Load Selected Session"):
                    st.session_state.messages = sessions[selected_session]['messages']
                    st.sidebar.success("Learning session loaded successfully!")
            else:
                st.sidebar.error("No valid learning sessions found in the uploaded file.")
        except Exception as e:
            st.sidebar.error(f"Error loading learning sessions: {str(e)}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about finance or request a lesson:"):
        st.session_state.messages.append({"role": "user", "content": f"{st.session_state.user_name}: {prompt}"})
        with st.chat_message("user"):
            st.markdown(f"{st.session_state.user_name}: {prompt}")

        topic_instruction = f"Provide education on {financial_topic}, focusing on {subtopic}. "
        ai_messages = [
            {"role": "system", "content": custom_instructions + topic_instruction},
            {"role": "system", "content": "Offer clear explanations, practical examples, and engage the user with questions to reinforce their understanding of financial concepts."},
        ] + st.session_state.messages

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in stream_response(ai_messages, model):
                if chunk:
                    if model.startswith("gpt-"):
                        full_response += chunk.choices[0].delta.content or ""
                    else:
                        full_response += chunk['message']['content']
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.05)
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

        _, prompt_tokens, completion_tokens = get_ai_response(ai_messages, model)
        st.session_state.token_count["prompt"] += prompt_tokens
        st.session_state.token_count["completion"] += completion_tokens

    st.sidebar.subheader("Token Usage")
    st.sidebar.write(f"Prompt tokens: {st.session_state.token_count['prompt']}")
    st.sidebar.write(f"Completion tokens: {st.session_state.token_count['completion']}")
    st.sidebar.write(f"Total tokens: {sum(st.session_state.token_count.values())}")

if __name__ == "__main__":
    main()
