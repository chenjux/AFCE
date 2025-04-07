import re, os, argparse
import pandas as pd
from datetime import datetime
from llm_api_utils import completion
from tqdm import tqdm
from dataloader import load_dataset

class SamplingDataProcessor:
    """Processor for confidence estimation using sampling-based strategy."""

    def __init__(self, dataset, model="claude-3-sonnet-20240229", temperature=0.7, num_samples=3):
        """
        Initialize the processor.

        Args:
            dataset: A pandas DataFrame with columns ['question', 'A', 'B', 'C', 'D', 'answer']
            model: The LLM to use
            temperature: Temperature setting for sampling
            num_samples: Number of samples to generate per question
        """
        self.dataset = dataset
        self.model = model
        self.temperature = temperature
        self.num_samples = num_samples
        self.correct_answers = dataset['answer'].tolist()

        self.llm_answers = []
        self.confidences = []

        self._process_dataset()
        self.dataframe = self._create_output_dataframe()

    def _generate_prompt(self, question, options):
        prompt = f"""Read the question, only provide your option letter and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true from 0 to 100.\n\nQuestion: {question}\n"""
        for key, value in options.items():
            prompt += f"{key}. {value}\n"
        prompt += "Answer:\nConfidence:"
        return prompt

    def _extract_result(self, response):
        lines = [line.strip() for line in response.split('\n')]
        answer = None
        confidence = None

        for line in lines:
            if not answer:
                match = re.match(r'^([A-Z])(?:[.\s]|$)', line)
                if match:
                    answer = match.group(1)
            if line.startswith("Confidence:"):
                match = re.search(r'\d+', line)
                if match:
                    confidence = float(match.group())

        if confidence is None and lines:
            match = re.search(r'\d+', lines[0])
            if match:
                confidence = float(match.group())

        return answer or 'N/A', confidence if confidence is not None else 90.0

    def _sample_responses(self, question, options):
        responses = []
        for _ in range(self.num_samples):
            prompt = self._generate_prompt(question, options)
            response = completion(prompt=prompt, model=self.model, temperature=self.temperature)
            responses.append(response)
        return responses

    def _aggregate_confidence(self, responses):
        answer_confidences = {}
        total_confidence = 0

        for r in responses:
            answer, conf = self._extract_result(r)
            if answer not in answer_confidences:
                answer_confidences[answer] = 0
            answer_confidences[answer] += conf
            total_confidence += conf

        if not answer_confidences or total_confidence == 0:
            return 'N/A', 0.0

        final_answer = max(answer_confidences, key=answer_confidences.get)
        final_confidence_score = answer_confidences[final_answer] / total_confidence
        return final_answer, final_confidence_score

    def _process_dataset(self):
        print(f'Processing dataset using Sampling method with {self.model}...')
        for _, row in tqdm(self.dataset.iterrows(), total=len(self.dataset)):
            question = row['question']
            options = {'A': row['A'], 'B': row['B'], 'C': row['C'], 'D': row['D']}
            try:
                responses = self._sample_responses(question, options)
                answer, confidence = self._aggregate_confidence(responses)
            except Exception as e:
                print(f"Error processing question: {e}")
                answer, confidence = 'error', 0.0

            self.llm_answers.append(answer)
            self.confidences.append(confidence)

    def _create_output_dataframe(self):
        num_questions = len(self.llm_answers)
        return pd.DataFrame({
            'question_id': [f'Q{i+1}' for i in range(num_questions)],
            'question': self.dataset['question'][:num_questions].tolist(),
            'llm_answer': self.llm_answers,
            'confidence': self.confidences,
            'correct_answer': self.correct_answers[:num_questions],
            'is_correct': [llm == cor for llm, cor in zip(self.llm_answers, self.correct_answers[:num_questions])],
            'model': self.model,
            'temperature': self.temperature,
            'num_samples': self.num_samples,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })


def run_experiment(input_paths, model, temperature, num_samples, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for path in input_paths:
        dataset_name = os.path.splitext(os.path.basename(path))[0]
        print(f"\n--- Running on dataset: {dataset_name} ---")
        df = load_dataset(path)

        processor = SamplingDataProcessor(
            dataset=df,
            model=model,
            temperature=temperature,
            num_samples=num_samples
        )

        output_df = processor.dataframe
        output_path = os.path.join(output_dir, f"{dataset_name}_{model}_sampling.csv")
        output_df.to_csv(output_path, index=False)
        print(f"Saved results to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run sampling-based confidence estimation across datasets.")
    parser.add_argument('--input_paths', nargs='+', required=True, help='List of CSV file paths.')
    parser.add_argument('--model', type=str, default='claude-3-sonnet-20240229', help='LLM model to use.')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature.')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to draw per question.')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results.')
    return parser.parse_args()


def main():
    args = parse_args()
    run_experiment(
        input_paths=args.input_paths,
        model=args.model,
        temperature=args.temperature,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()