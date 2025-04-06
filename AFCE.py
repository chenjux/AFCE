from llm_api_utils import completion
from dataloader import load_dataset
import pandas as pd
import re, os, random, sys, argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm

class QuizLikeDataProcessor:
    def __init__(self, dataset, experiment_type, model,
                 questions_per_quiz):
        self.model = model
        self.dataset = dataset
        self.questions_per_quiz = questions_per_quiz
        self.experiment_type = experiment_type
        self.correct_answers_list = dataset['answer'].tolist()
        self.max_questions = self.get_max_num_questions()
        self.quizzes = self.get_quizzes()
        # Create prompts and extract responses based on the experiment type
        self.prompt_quiz_list = self.create_prompts()
        self.llm_answers = []
        self.estimates = []

        self.process_dataset()

        # Create DataFrame to store estimates and LLM answers
        self.dataframe = self.create_dataframe()

    def get_max_num_questions(self):
        return (len(self.dataset['question']) // self.questions_per_quiz) * self.questions_per_quiz

    def get_quizzes(self):
        num_questions = min(self.max_questions, len(self.dataset['question']))
        selected_questions = {
            'question': self.dataset['question'][:num_questions].tolist(),
            'A': self.dataset['A'][:num_questions].tolist(),
            'B': self.dataset['B'][:num_questions].tolist(),
            'C': self.dataset['C'][:num_questions].tolist(),
            'D': self.dataset['D'][:num_questions].tolist(),
        }
        quizzes = [
            {
                'question': selected_questions['question'][i:i + self.questions_per_quiz],
                'A': selected_questions['A'][i:i + self.questions_per_quiz],
                'B': selected_questions['B'][i:i + self.questions_per_quiz],
                'C': selected_questions['C'][i:i + self.questions_per_quiz],
                'D': selected_questions['D'][i:i + self.questions_per_quiz],
            }
            for i in range(0, num_questions, self.questions_per_quiz)
        ]
        return quizzes

    def create_prompts(self):
        if self.experiment_type == 'answer_only':
            return self.prompt_create_quiz_answer_only()
        elif self.experiment_type == 'estimate_only':
            return self.prompt_create_quiz_estimate_only()
        else:
            raise ValueError(f"Unknown experiment type: {self.experiment_type}")

    def prompt_create_quiz_answer_only(self):
        front_prompt = "Please answer the following questions by selecting only the option letter. Do not provide any explanation.\nQuestions:\n"
        back_prompt = "Now, please answer these questions following the format:\n" + "\n".join(
            [f"{i}. <Only Option Letter>" for i in range(1, self.questions_per_quiz + 1)]) + "\n"
        return self._create_quiz_prompts(front_prompt, back_prompt)

    def prompt_create_quiz_estimate_only(self):
        front_prompt = f"Read the question, estimate how many questions you can answer correctly (choose one number from 0-{str(self.questions_per_quiz)}). Do not provide any explanation.\nQuestions:\n"
        back_prompt = f"""Estimate how many questions you can answer correctly: <only choose one number from 0-{str(self.questions_per_quiz)}>."""

        return self._create_quiz_prompts(front_prompt, back_prompt)

    def _create_quiz_prompts(self, front_prompt, back_prompt):
        quizzes = []
        for quiz in self.quizzes:
            questions = []
            for i in range(len(quiz['question'])):
                question = quiz['question'][i]

                # Format the question and choices
                formatted_question = f"{i + 1}. {question}\n"
                formatted_question += f"   A. {quiz['A'][i]}\n"
                formatted_question += f"   B. {quiz['B'][i]}\n"
                formatted_question += f"   C. {quiz['C'][i]}\n"
                formatted_question += f"   D. {quiz['D'][i]}\n"

                questions.append(formatted_question)

            all_questions = "\n".join(questions)
            full_prompt = front_prompt + all_questions + back_prompt
            quizzes.append(full_prompt)

        return quizzes

    def extract_answers_only(self, response):
        lines = response.strip().split('\n')
        answers = []

        for line in lines:
            match = re.match(r'^\d+\.(.*)$', line.strip())
            if match:
                answer = match.group(1).strip()
                answers.append(answer if answer in ['A', 'B', 'C', 'D'] else 'C')  # Default to 'C' if invalid answer

        return answers

    def extract_estimates_only(self, response):
        estimate_match = re.search(r'(\d+)', response)
        return int(estimate_match.group(1)) if estimate_match else 0  # Default to 0 if no estimate is found

    def process_dataset(self):
        print(f'Processing dataset for {self.experiment_type}...')
        for complete_prompt in self.prompt_quiz_list:
            print(complete_prompt)
            response = completion(complete_prompt, self.model)

            if self.experiment_type == 'answer_only':
                answers = self.extract_answers_only(response)
                self.llm_answers.extend(answers)
            elif self.experiment_type == 'estimate_only':
                estimate = self.extract_estimates_only(response)
                self.estimates.append(estimate)

    def calculate_quiz_accuracy(self, llm_answers, correct_answers):
        """Calculate accuracy for each quiz as the proportion of correct answers."""
        correct_count = sum([1 for llm, correct in zip(llm_answers, correct_answers) if llm == correct])
        return correct_count / len(correct_answers)

    def create_dataframe(self):
        num_questions = self.max_questions
        num_groups = num_questions // self.questions_per_quiz

        # Ensure all lists are the same length
        llm_answers_filled = (self.llm_answers + ['N/A'] * num_questions)[:num_questions]
        correct_answers_filled = (self.correct_answers_list + ['N/A'] * num_questions)[:num_questions]
        questions = self.dataset['question'][:num_questions].tolist()

        # Pack every 10 answers into a list
        def pack_into_lists(data, size=self.questions_per_quiz):
            return [data[i:i + size] for i in range(0, len(data), size)]

        packed_llm_answers = pack_into_lists(llm_answers_filled) if self.experiment_type != 'estimate_only' else [
            ['N/A'] * self.questions_per_quiz for _ in range(num_groups)
        ]
        packed_correct_answers = pack_into_lists(correct_answers_filled)
        packed_questions = pack_into_lists(questions)

        # Calculate quiz accuracy for each quiz
        quiz_accuracies = [self.calculate_quiz_accuracy(llm_ans, correct_ans)
                           for llm_ans, correct_ans in zip(packed_llm_answers, packed_correct_answers)]

        # Adjust estimates list if experiment_type is 'answer_only'
        if self.experiment_type == 'answer_only':
            estimates_filled = [None] * num_groups
            llm_confidences = [None] * num_groups
        else:
            estimates_filled = (self.estimates + [None] * num_groups)[:num_groups]
            llm_confidences = [i / self.questions_per_quiz if i is not None else None for i in estimates_filled]

        data = {
            'quiz_id': [f'Quiz_{i + 1}' for i in range(num_groups)],
            'question': packed_questions,
            'llm_answers': packed_llm_answers,
            'correct_answers': packed_correct_answers,
            'estimates': estimates_filled,
            'llm_confidences': llm_confidences,
            'quiz_accuracy': quiz_accuracies,
            'parameters': self.experiment_type,
            'model_name': self.model
        }

        return pd.DataFrame(data)

def combine_dataframes(answer_only_df, estimate_only_df):
    combined_df = answer_only_df.copy()
    combined_df['estimates'] = estimate_only_df['estimates']
    combined_df['llm_confidences'] = estimate_only_df['llm_confidences']
    combined_df['parameters'] = 'AFCE'
    return combined_df


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run LLM experiments')
    parser.add_argument('--datasets',
                        nargs='+',
                        default=['all'],
                        help='select datasets(default: all)')
    parser.add_argument('--questions_per_quiz',
                        type=int,
                        default=10,
                        help='num of questions per quiz(default: 10)')
    parser.add_argument('--output_dir',
                       default='results',
                       help='output path (default: results)')
    parser.add_argument('--model',
                       default='gpt-4o',
                       help='model (default: gpt-4o)')
    return parser.parse_args()


# 数据集加载模块
def load_datasets():
    return {
        "college_physics": load_dataset("college_physics", sample_size=100),
        "high_school_physics": load_dataset("high_school_physics", sample_size=150),
        "college_biology": load_dataset("college_biology", sample_size=140),
        "high_school_biology": load_dataset("high_school_biology", sample_size=310),
        "college_chemistry": load_dataset("college_chemistry", sample_size=100),
        "high_school_chemistry": load_dataset("high_school_chemistry", sample_size=200),
        "gpqa_physics": load_dataset("gpqa_physics", sample_size=180),
        "gpqa_chemistry": load_dataset("gpqa_chemistry", sample_size=180),
        "gpqa_biology": load_dataset("gpqa_biology", sample_size=70),
    }


def main():
    global_results = []
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

    for ds_name, dataset in progress_bar:
        progress_bar.set_postfix({'dataset': ds_name})
        try:
            # Sanitize dataset name for filename
            sanitized_name = ds_name.replace(" ", "_")

            # Run answer-only experiment
            answer_processor = QuizLikeDataProcessor(
                dataset=dataset,
                experiment_type='answer_only',
                questions_per_quiz=args.questions_per_quiz,
                model=args.model,
            )

            # Run estimate-only experiment
            estimate_processor = QuizLikeDataProcessor(
                dataset=dataset,
                experiment_type='estimate_only',
                questions_per_quiz=args.questions_per_quiz,
                model = args.model
            )

            # Combine results
            combined_df = combine_dataframes(
                answer_processor.dataframe,
                estimate_processor.dataframe
            )
            combined_df['dataset_name'] = ds_name

            # Save individual results
            filename = f"{sanitized_name}_q{args.questions_per_quiz}_{timestamp}.csv"
            output_path = os.path.join(args.output_dir, filename)
            combined_df.to_csv(output_path, index=False)
            tqdm.write(f"✓ Successfully saved: {output_path}")

            global_results.append(combined_df)

        except Exception as e:
            tqdm.write(f"✕ Processing failed [{ds_name}]: {str(e)}")
            continue
if __name__ == "__main__":
    main()