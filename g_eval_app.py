import streamlit as st
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import statistics
from openai import AsyncOpenAI
import requests
import os
from dotenv import load_dotenv
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import math
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationDimension(Enum):
    """G-Eval dimensions from the documentation"""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    HALLUCINATION = "hallucination"
    TONE = "tone"

@dataclass
class ModelConfig:
    name: str
    api_key: str
    model_name: str
    base_url: Optional[str] = None

@dataclass
class ModelEvaluation:
    model_name: str
    dimension: EvaluationDimension
    score: float
    reasoning: str
    timestamp: datetime

class GEvalJudge:
    """
    G-Eval implementation following the original paper methodology
    Uses multiple LLMs as judges with chain-of-thought evaluation
    """
    
    def __init__(self):
        self.models = [
            ModelConfig(
                name="OpenAI GPT-4",
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="gpt-4-turbo-preview",
                base_url=None
            ),
            ModelConfig(
                name="Anthropic Claude",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                model_name="claude-3-sonnet-20240229",
                base_url="https://api.anthropic.com/v1"
            )
        ]
        self.clients = self._initialize_clients()
        self.evaluation_criteria = self._setup_criteria()
        
    def _initialize_clients(self) -> Dict[str, AsyncOpenAI]:
        """Initialize LLM clients"""
        clients = {}
        for model in self.models:
            if model.base_url:
                clients[model.name] = AsyncOpenAI(
                    api_key=model.api_key,
                    base_url=model.base_url
                )
            else:
                clients[model.name] = AsyncOpenAI(api_key=model.api_key)
        return clients
    
    def _setup_criteria(self) -> Dict[EvaluationDimension, str]:
        """G-Eval prompts following the documentation methodology"""
        return {
            EvaluationDimension.ACCURACY: """
You will evaluate the factual accuracy of an answer.

Evaluation Steps:
1. Check whether facts in the answer contradict the context or known information
2. Verify if key factual claims are supported and accurate
3. Assess precision and completeness of factual information
4. Heavily penalize fabricated or unsupported claims
5. Consider if omissions significantly impact factual accuracy

Question: {question}
Context: {context}
Answer: {answer}

Rate accuracy on a scale of 1-5:
1 = Completely inaccurate, major factual errors
2 = Mostly inaccurate with some correct elements
3 = Mixed accuracy, significant errors present
4 = Mostly accurate with minor errors
5 = Completely accurate, all facts correct

Score: [Provide score 1-5]
Reasoning: [Detailed step-by-step reasoning]
""",
            EvaluationDimension.COMPLETENESS: """
You will evaluate how thoroughly the answer addresses the question.

Evaluation Steps:
1. Identify all key aspects the question is asking for
2. Check if the answer addresses each identified aspect
3. Assess depth and thoroughness of coverage
4. Evaluate whether important details are included or omitted
5. Consider if the answer provides sufficient information

Question: {question}
Context: {context}
Answer: {answer}

Rate completeness on a scale of 1-5:
1 = Severely incomplete, missing most key components
2 = Incomplete, missing several important aspects
3 = Partially complete, covers main points but lacks details
4 = Mostly complete, minor gaps in coverage
5 = Comprehensive, thoroughly addresses all aspects

Score: [Provide score 1-5]
Reasoning: [Detailed step-by-step reasoning]
""",
            EvaluationDimension.HALLUCINATION: """
You will evaluate the answer for fabricated or unsupported information.

Evaluation Steps:
1. Identify all factual claims and assertions in the answer
2. Check each claim against the provided context and known information
3. Look for fabricated names, dates, statistics, or details
4. Assess whether claims are properly grounded in available information

Question: {question}
Context: {context}
Answer: {answer}

Rate hallucination on a scale of 1-5:
1 = Severe hallucination, significant fabricated information
2 = Major hallucinations, multiple unsupported claims
3 = Some hallucinations, several questionable statements
4 = Minor hallucinations, mostly grounded with few issues
5 = No hallucination, all information well-grounded

Provide your evaluation in the following format:
Score: [Provide score 1-5]
Reasoning: 
1. Conclusion: [Write a one-sentence conclusion about whether the answer contains hallucinations]
2. Key Findings: [List only the specific claims that are hallucinations or unsupported, if any]
3. Evidence: [If hallucinations were found, provide specific evidence from the context that contradicts or fails to support these claims]
""",
            EvaluationDimension.TONE: """
You will evaluate the appropriateness of tone and communication style.

Evaluation Steps:
1. Assess whether tone matches the context and expected audience
2. Evaluate if language is professional and appropriate for the domain
3. Check for clarity, respectfulness, and accessibility
4. Identify any inappropriate casual language or unclear expressions
5. Consider overall coherence and readability

Question: {question}
Context: {context}
Answer: {answer}

Rate tone on a scale of 1-5:
1 = Highly inappropriate, unprofessional or unsuitable
2 = Mostly inappropriate, significant tone issues
3 = Mixed, some appropriate elements but notable problems
4 = Mostly appropriate, minor tone improvements needed
5 = Perfect tone, professional, clear, and contextually appropriate

Score: [Provide score 1-5]
Reasoning: [Detailed step-by-step reasoning]
"""
        }
    
    async def evaluate_single_model(
        self, 
        model_config: ModelConfig, 
        question: str, 
        answer: str, 
        context: str, 
        dimension: EvaluationDimension
    ) -> Optional[ModelEvaluation]:
        """G-Eval single model evaluation"""
        try:
            client = self.clients[model_config.name]
            prompt = self.evaluation_criteria[dimension].format(
                question=question,
                answer=answer,
                context=context
            )
            
            response = await client.chat.completions.create(
                model=model_config.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. Follow G-Eval methodology and provide step-by-step reasoning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract score using regex (G-Eval methodology)
            import re
            score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', content, re.IGNORECASE)
            reasoning_match = re.search(r'Reasoning:\s*(.+?)(?=\n\n|\Z)', content, re.IGNORECASE | re.DOTALL)
            
            if score_match:
                raw_score = float(score_match.group(1))
                # G-Eval normalization: convert 1-5 to 0-1
                normalized_score = (raw_score - 1) / 4
            else:
                # Fallback extraction
                numbers = re.findall(r'\b([1-5](?:\.\d+)?)\b', content)
                if numbers:
                    raw_score = float(numbers[0])
                    normalized_score = (raw_score - 1) / 4
                else:
                    raise ValueError("Could not extract score")
            
            reasoning = reasoning_match.group(1).strip() if reasoning_match else content
            
            return ModelEvaluation(
                model_name=model_config.name,
                dimension=dimension,
                score=normalized_score,
                reasoning=reasoning,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            st.error(f"Error with {model_config.name} for {dimension.value}: {str(e)}")
            return None
    
    async def evaluate_ensemble(
        self, 
        question: str, 
        answer: str, 
        context: str = ""
    ) -> Dict[str, any]:
        """G-Eval ensemble evaluation across all dimensions"""
        
        results = {}
        all_evaluations = []
        
        # Evaluate each dimension with all models
        for dimension in EvaluationDimension:
            st.text(f"Evaluating {dimension.value}...")
            
            # Get evaluations from all models for this dimension
            tasks = []
            for model in self.models:
                tasks.append(self.evaluate_single_model(model, question, answer, context, dimension))
            
            model_evaluations = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful evaluations
            valid_evaluations = [
                eval for eval in model_evaluations 
                if isinstance(eval, ModelEvaluation)
            ]
            
            if valid_evaluations:
                # Calculate aggregate score for this dimension (G-Eval methodology)
                scores = [eval.score for eval in valid_evaluations]
                aggregate_score = statistics.mean(scores)
                
                results[dimension.value] = {
                    'aggregate_score': aggregate_score,
                    'individual_scores': {eval.model_name: eval.score for eval in valid_evaluations},
                    'reasoning': {eval.model_name: eval.reasoning for eval in valid_evaluations}
                }
                
                all_evaluations.extend(valid_evaluations)
        
        # Calculate overall quality score (G-Eval final score)
        if results:
            overall_score = statistics.mean([dim_result['aggregate_score'] for dim_result in results.values()])
            results['overall_quality'] = overall_score
        
        return results

def main():
    st.set_page_config(
        page_title="G-Eval: LLM Judge",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("G-Eval: LLM Judge")
    st.markdown("Multiple LLM judges evaluating agent responses using G-Eval methodology")
    
    # Initialize G-Eval judge
    if 'judge' not in st.session_state:
        st.session_state.judge = GEvalJudge()
    
    st.markdown("---")
    
    # File Upload Section
    st.subheader("Ground Truth Data")
    uploaded_file = st.file_uploader("Upload Excel file with questions and answers", type=['xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            # Read the Excel file, skipping the separator row
            df = pd.read_excel(uploaded_file, skiprows=[1])
            
            # Clean the dataframe
            df = df.replace({np.nan: None})  # Replace NaN with None
            df = df.dropna(how='all')  # Remove completely empty rows
            
            # Clean up the columns
            df['Question'] = df['Question'].apply(lambda x: str(x).strip() if pd.notnull(x) else None)
            df['Answer'] = df['Answer'].apply(lambda x: str(x).strip() if pd.notnull(x) else None)
            
            # Remove rows where either Question or Answer is None/empty
            df = df[df['Question'].notna() & df['Answer'].notna()]
            df = df[df['Question'].str.len() > 0]
            df = df[df['Answer'].str.len() > 0]
            
            # Reset index after cleaning
            df = df.reset_index(drop=True)
            
            # Display the cleaned data
            st.success("Successfully loaded and cleaned the file!")
            st.write("Preview of Questions and Answers:")
            st.dataframe(df)
            
            # Show total number of questions
            st.info(f"Total number of questions to evaluate: {len(df)}")
            
            # Agent Configuration
            st.subheader("Agent API Configuration")
            col1, col2 = st.columns(2)
            with col1:
                api_url = st.text_input("Agent API URL", value=os.getenv("API_URL", ""))
                api_key = st.text_input("Agent API Key", type="password", value=os.getenv("API_KEY", ""))
            with col2:
                conversation_id = st.text_input("Conversation ID", value="geval_test_123")
                timeout = st.number_input("Timeout (seconds)", value=30, min_value=5)
                max_retries = st.number_input("Max Retries", value=3, min_value=1)
            
            # Execute G-Eval
            if st.button("🚀 Start Evaluation", type="primary"):
                if not api_url or not api_key:
                    st.error("Please configure your agent API")
                else:
                    # Initialize metrics
                    overall_metrics = {
                        'accuracy': [],
                        'completeness': [],
                        'hallucination': [],
                        'tone': [],
                        'overall_quality': []
                    }
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process each question
                    for index, row in df.iterrows():
                        progress = (index + 1) / len(df)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing question {index + 1} of {len(df)}")
                        
                        try:
                            # Call agent API
                            headers = {
                                "x-functions-key": api_key,
                                "Content-Type": "application/json",
                            }
                            payload = {
                                "conversation": row['Question'],
                                "conversation_id": f"{conversation_id}_{index}"
                            }
                            
                            response = requests.post(api_url, json=payload, headers=headers, timeout=timeout)
                            
                            if response.status_code == 200:
                                agent_answer = response.json().get("answer") or response.text
                                
                                # Run G-Eval
                                geval_results = asyncio.run(
                                    st.session_state.judge.evaluate_ensemble(
                                        row['Question'],
                                        agent_answer,
                                        ""  # No context
                                    )
                                )
                                
                                if geval_results:
                                    # Store metrics
                                    for metric in overall_metrics.keys():
                                        if metric in geval_results:
                                            if isinstance(geval_results[metric], dict):
                                                overall_metrics[metric].append(geval_results[metric]['aggregate_score'])
                                            else:
                                                overall_metrics[metric].append(geval_results[metric])
                                    
                                    # Create expandable section for each question
                                    with st.expander(f"Question {index + 1}: {row['Question'][:100]}..."):
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.write("**Ground Truth Answer:**")
                                            st.write(row['Answer'])
                                            st.write("**Agent Answer:**")
                                            st.write(agent_answer)
                                        
                                        with col2:
                                            # Create bar chart for dimension scores
                                            dimensions = ['accuracy', 'completeness', 'hallucination', 'tone']
                                            scores = [geval_results[dim]['aggregate_score'] for dim in dimensions if dim in geval_results]
                                            
                                            fig_dimensions = go.Figure(go.Bar(
                                                x=dimensions,
                                                y=scores,
                                                marker_color='rgb(51, 153, 255)'
                                            ))
                                            fig_dimensions.update_layout(
                                                title="Dimension Scores",
                                                yaxis=dict(range=[0, 1], title="Score"),
                                                xaxis=dict(title="Dimensions"),
                                                height=300
                                            )
                                            st.plotly_chart(fig_dimensions, use_container_width=True)
                                        
                                        # Display model reasonings
                                        st.write("**Evaluation Details:**")
                                        for dim in dimensions:
                                            if dim in geval_results:
                                                for model_name, reasoning in geval_results[dim]['reasoning'].items():
                                                    with st.expander(f"{dim.title()} - {model_name} Reasoning"):
                                                        st.write(reasoning)
                            else:
                                st.error(f"Error with question {index + 1}: API returned status code {response.status_code}")
                                
                        except Exception as e:
                            st.error(f"Error processing question {index + 1}: {str(e)}")
                    
                    # Display overall metrics
                    st.subheader("Overall Evaluation Results")
                    
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    metrics = {
                        'accuracy': 'Accuracy',
                        'completeness': 'Completeness',
                        'hallucination': 'Hallucination',
                        'tone': 'Tone'
                    }
                    
                    for (metric, title), col in zip(metrics.items(), [col1, col2, col3, col4]):
                        with col:
                            if overall_metrics[metric]:
                                avg_score = sum(overall_metrics[metric]) / len(overall_metrics[metric])
                                st.metric(title, f"{avg_score:.3f}")
                    
                    # Overall quality trend chart
                    st.subheader("Overall Quality Trend")
                    fig_overall = go.Figure(go.Scatter(
                        x=list(range(1, len(overall_metrics['overall_quality']) + 1)),
                        y=overall_metrics['overall_quality'],
                        mode='lines+markers'
                    ))
                    fig_overall.update_layout(
                        xaxis_title="Question Number",
                        yaxis_title="Quality Score",
                        yaxis=dict(range=[0, 1])
                    )
                    st.plotly_chart(fig_overall, use_container_width=True)
                    
                    # Save results
                    results_df = pd.DataFrame(overall_metrics)
                    st.download_button(
                        label="📥 Download Evaluation Results",
                        data=results_df.to_csv(index=False).encode('utf-8'),
                        file_name='evaluation_results.csv',
                        mime='text/csv'
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()