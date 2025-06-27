# G-Eval: LLM Judge

A Streamlit application for evaluating LLM responses using multiple judges (GPT-4 and Claude) based on the G-Eval methodology.

## Features

- Upload Excel files with questions and answers for evaluation
- Evaluate responses across multiple dimensions:
  - Accuracy
  - Completeness
  - Hallucination
  - Tone
- Visual analytics with charts and graphs
- Detailed reasoning from multiple LLM judges
- Downloadable evaluation results

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/g-eval.git
cd g-eval
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```plaintext
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
API_URL=your_agent_api_url
API_KEY=your_agent_api_key
```

4. Run the application:
```bash
streamlit run g_eval_app.py
```

## Excel File Format

The input Excel file should have the following format:

| Question | Answer |
|----------|---------|
| Q1       | A1      |
| Q2       | A2      |

- First row: Headers ("Question" and "Answer")
- Second row: Separator (optional)
- Following rows: Questions and their corresponding answers

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

MIT License 
