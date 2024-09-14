# Financial Literacy Educator: Interactive Modules on Budgeting, Investing, and Economics

Financial Literacy Educator is a Streamlit-based web application that provides interactive education on various financial topics, including budgeting, investing, and economics. This tool uses various language models to offer explanations, practical examples, and engage users in learning about personal finance and economic concepts.

## Features

- Interactive chat interface for asking questions and receiving financial education
- Support for multiple AI models, including OpenAI's GPT models and Ollama's local models
- Customizable financial topics and subtopics
- Dark/Light theme toggle
- Learning session saving and loading functionality
- Token usage tracking

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/lalomorales22/FinancialLiteracyEducator-Streamlit100.git
   cd financial-literacy-educator
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY='your-api-key-here'
   ```

4. (Optional) If you want to use Ollama models, make sure you have Ollama installed and running on your system.

## Usage

1. Run the Streamlit app:
   ```
   streamlit run financial_literacy_educator.py
   ```

2. Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Enter your name, select the financial topic and subtopic, and start learning about finance!

## Customization

- You can modify the `FINANCIAL_TOPICS` and `SUBTOPICS` dictionaries in the code to add or remove financial topics and subtopics.
- The custom instructions for the AI can be adjusted in the sidebar of the application.

## Contributing

Contributions to improve the Financial Literacy Educator are welcome! Please feel free to submit pull requests or open issues to discuss potential enhancements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
