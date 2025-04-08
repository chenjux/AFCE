import pandas as pd
import re
from datetime import datetime
from utils.llm_api_utils import completion

class OpenEndedProcessor:
    def __init__(self, dataset, model="llama3-70b-8192", temperature=0):
        self.dataset = dataset
        self.model = model
        self.temperature = temperature
        self.responses = []
        self.correct_flags = []
        self.confidences = []
        self.correct_answers = dataset['answer'].tolist()

        self._process_dataset()
        self.dataframe = self._create_output_dataframe()

    def _generate_prompt(self, question):
        return (
            f"Read the question, only provide your answer and your confidence in this answer. "
            f"Note: The confidence indicates how likely you think your answer is true from 0 to 100.\n\n"
            f"Question: {question}\nAnswer:\nConfidence:"
        )

    def _ask_question(self, question):
        prompt = self._generate_prompt(question)
        response = completion(prompt=prompt, model=self.model, temperature=self.temperature)
        return response.text if hasattr(response, 'text') else response

    def _check_response(self, response, correct_answer):
        prompt = (
            f"Help me extract the confidence from the text. Then check the answer. "
            f"Is the following response correct? Response: '{response}'. Correct Answer: '{correct_answer}'. "
            f"Provide response in following format. Make it as concise as possible:\n"
            f"True or False:\nConfidence:"
        )
        reply = completion(prompt=prompt, model=self.model, temperature=self.temperature)
        result = reply.text if hasattr(reply, 'text') else reply
        result = result.strip().lower()

        is_correct = False
        confidence = 0.0

        for line in result.split('\n'):
            if 'true' in line:
                is_correct = True
            elif 'false' in line:
                is_correct = False
            if 'confidence:' in line:
                match = re.search(r'\d+(\.\d+)?', line)
                if match:
                    val = float(match.group())
                    confidence = val / 100 if val > 1 else val

        return is_correct, max(0.0, min(1.0, confidence))

    def _process_dataset(self):
        for _, row in self.dataset.iterrows():
            question = row['question']
            correct_answer = row['answer']
            try:
                response = self._ask_question(question)
                is_correct, confidence = self._check_response(response, correct_answer)
            except Exception as e:
                print(f"Error processing: {e}")
                response, is_correct, confidence = 'error', False, 0.0

            self.responses.append(response)
            self.correct_flags.append(is_correct)
            self.confidences.append(confidence)

    def _create_output_dataframe(self):
        num_questions = len(self.responses)
        return pd.DataFrame({
            'question_id': [f'Q{i+1}' for i in range(num_questions)],
            'question': self.dataset['question'].tolist(),
            'llm_answer': self.responses,
            'confidence': self.confidences,
            'correct_answer': self.correct_answers,
            'is_correct': self.correct_flags,
            'model': self.model,
            'temperature': self.temperature,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })