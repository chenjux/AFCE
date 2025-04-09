from utils.llm_api_utils import completion
from code.utils.dataloader import load_dataset
import pandas as pd
import re
import os
import argparse
from datetime import datetime
from tqdm import tqdm

class TopKDataProcessor:
    """Processor for Top-k confidence estimation experiments."""

    def __init__(self, dataset, model):
        """
        Initialize the Top-k data processor.

        Args:
            dataset: Input dataset containing questions and answers
            model: Name of the LLM model to use
        """
        self.model = model
        self.dataset = dataset
        self.correct_answers = dataset['answer'].tolist()
        self.max_questions = self._get_max_questions()
        self.quizzes = self._prepare_quizzes()

        # Response storage
        self.llm_answers = []
        self.confidences = []

        # Process the dataset
        self._process_dataset()
        self.dataframe = self._create_output_dataframe()

    def _get_max_questions(self):
        """Calculate maximum number of questions to process."""
        return (len(self.dataset['question']) // 10) * 10

    def _prepare_quizzes(self, questions_per_quiz=1):
        """Organize questions into quiz format."""
        num_questions = min(self.max_questions, len(self.dataset['question']))

        quiz_data = {
            'question': self.dataset['question'][:num_questions].tolist(),
            'A': self.dataset['A'][:num_questions].tolist(),
            'B': self.dataset['B'][:num_questions].tolist(),
            'C': self.dataset['C'][:num_questions].tolist(),
            'D': self.dataset['D'][:num_questions].tolist()
        }

        return [
            {
                'question': quiz_data['question'][i:i + questions_per_quiz],
                'choices': {
                    'A': quiz_data['A'][i:i + questions_per_quiz],
                    'B': quiz_data['B'][i:i + questions_per_quiz],
                    'C': quiz_data['C'][i:i + questions_per_quiz],
                    'D': quiz_data['D'][i:i + questions_per_quiz]
                }
            }
            for i in range(0, num_questions, questions_per_quiz)
        ]

    def _generate_prompts(self):
        """Generate prompts for top-k confidence estimation."""
        instruction = """Provide your k best guesses and confidence scores (0-100%) for the following question. 
        Use exactly this format:
        G1: <option letter> 
        P1: <confidence percentage>
        ...
        G4: <option letter>
        P4: <confidence percentage>
        Question: """

        return self._format_quiz_prompts(instruction, "")

    def _format_quiz_prompts(self, prefix, suffix):
        """Format complete quiz prompts."""
        prompts = []
        for quiz in self.quizzes:
            questions = []
            for i in range(len(quiz['question'])):
                q = quiz['question'][i]
                choices = quiz['choices']

                formatted = f"{i + 1}. {q}\n"
                formatted += f"   A. {choices['A'][i]}\n"
                formatted += f"   B. {choices['B'][i]}\n"
                formatted += f"   C. {choices['C'][i]}\n"
                formatted += f"   D. {choices['D'][i]}\n"

                questions.append(formatted)

            full_prompt = prefix + "\n".join(questions) + suffix
            prompts.append(full_prompt)

        return prompts

    def _extract_top_k_response(self, response):
        """Extract answers and confidences from LLM response."""
        response = response.strip().rstrip('.')
        lines = [line.strip() for line in response.splitlines() if line.strip()]

        answers = {}
        for line in lines:
            if line.startswith(('G', 'P')) and ':' in line:
                key, value = line.split(':', 1)
                answers[key.strip()] = value.strip()

        primary_answer = answers.get('G1', 'error')
        primary_confidence = answers.get('P1', '101%').replace('%', '')

        return primary_answer, primary_confidence

    def _process_dataset(self):
        """Process all quizzes through the LLM."""
        print(f'Processing dataset using Top-k method with {self.model}...')

        for prompt in tqdm(self._generate_prompts(), desc="Processing quizzes"):
            try:
                response = completion(prompt, self.model)
                answer, confidence = self._extract_top_k_response(response)
                self.llm_answers.append(answer)
                self.confidences.append(confidence)

            except Exception as e:
                print(f"Error processing quiz: {str(e)}")
                self.llm_answers.append('error')
                self.confidences.append('0')

    def _create_output_dataframe(self):
        """Create final output dataframe with results."""
        num_questions = min(len(self.llm_answers), self.max_questions)

        data = {
            'question_id': [f'Q{i + 1}' for i in range(num_questions)],
            'question': self.dataset['question'][:num_questions].tolist(),
            'llm_answer': self.llm_answers[:num_questions],
            'confidence': [float(c) / 100 if c.replace('.', '', 1).isdigit() else 0
                           for c in self.confidences[:num_questions]],
            'correct_answer': self.correct_answers[:num_questions],
            'is_correct': [llm == cor for llm, cor in
                           zip(self.llm_answers[:num_questions], self.correct_answers[:num_questions])],
            'model': self.model,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return pd.DataFrame(data)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Top-k confidence estimation experiments')
    parser.add_argument('--datasets', nargs='+', default=['all'],
                        help='Datasets to process (default: all)')
    parser.add_argument('--model', default='gpt-4o',
                        help='LLM model to use')
    parser.add_argument('--output_dir', default='results',
                        help='Output directory')
    return parser.parse_args()



def main():
    """Main execution function."""
    args = parse_arguments()
    dataset_map = load_datasets()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Get timestamp for all files in this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Validate dataset selection
    selected_datasets = []
    if 'all' in args.datasets:
        selected_datasets = list(dataset_map.items())
    else:
        selected_datasets = [(name, df) for name, df in dataset_map.items()
                             if name in args.datasets]

    if not selected_datasets:
        raise ValueError("No valid datasets selected. Check your dataset names.")

    # Process datasets
    progress_bar = tqdm(selected_datasets, desc="Processing datasets")
    global_results = []

    for ds_name, dataset in progress_bar:
        progress_bar.set_postfix({'dataset': ds_name})

        try:
            # Sanitize dataset name for filename
            sanitized_name = ds_name.replace(" ", "_")

            # Process dataset with TopK method
            processor = TopKDataProcessor(
                dataset=dataset,
                model=args.model
            )

            # Save individual results
            filename = f"{sanitized_name}_topk_{args.model}_{timestamp}.csv"
            output_path = os.path.join(args.output_dir, filename)
            processor.dataframe.to_csv(output_path, index=False)
            tqdm.write(f"✓ Successfully saved: {output_path}")

            global_results.append(processor.dataframe)

        except Exception as e:
            tqdm.write(f"✕ Processing failed [{ds_name}]: {str(e)}")
            continue


    print("\nProcessing completed successfully!")


if __name__ == "__main__":
    main()