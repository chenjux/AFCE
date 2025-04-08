import re
import argparse
import os
import pandas as pd
from dataloader import load_datasets
from llm_api_utils import completion

class OverplacementProcessor:
    def __init__(self, dataset, experiment_type=None, expertise=None, subject=None, model=None):
        self.model = model
        self.dataset = dataset
        self.experiment_type = experiment_type
        self.correct_answers_list = dataset['answer'].tolist()
        self.max_questions = self.get_max_num_questions()
        self.expertise = expertise
        self.subject = subject
        self.quizzes = self.get_quizzes()

        # Create prompts and extract responses based on the experiment type
        self.prompt_quiz_list = self.create_prompts()
        self.llm_answers = []
        self.estimates = []

        self.process_dataset()

        # Create DataFrame to store estimates and LLM answers
        self.dataframe = self.create_dataframe()

    def get_max_num_questions(self):
        return (len(self.dataset['question']) // 10) * 10

    def get_quizzes(self, questions_per_quiz=10):
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
                'question': selected_questions['question'][i:i + questions_per_quiz],
                'A': selected_questions['A'][i:i + questions_per_quiz],
                'B': selected_questions['B'][i:i + questions_per_quiz],
                'C': selected_questions['C'][i:i + questions_per_quiz],
                'D': selected_questions['D'][i:i + questions_per_quiz],
            }
            for i in range(0, num_questions, questions_per_quiz)
        ]
        return quizzes

    def prompt_create_quiz_expertise_answer_only_role_play_true(self):  # you+subject+expertise + answer
        front_prompt = f"""You are an {self.expertise} within {self.subject}. Please answer the following questions by selecting only the option letter. Do not provide any explanation. 
Questions:.\n"""
        back_prompt = "Now, please answer these questions following the format:\n" + "\n".join(
            [f"{i}. <Only Option Letter>" for i in range(1, 11)]) + "\n"
        return self._create_quiz_prompts(front_prompt, back_prompt)

    def prompt_create_quiz_expertise_answer_only_random_person(self):  # you random person+ answer
        front_prompt = f"""You are a randomly chosen person. Please answer the following questions by selecting only the option letter. Do not provide any explanation. 
Questions:\n"""

        back_prompt = "Now, please answer these questions following the format:\n" + "\n".join(
            [f"{i}. <Only Option Letter>" for i in range(1, 11)]) + "\n"
        return self._create_quiz_prompts(front_prompt, back_prompt)

    def prompt_create_quiz_expertise_estimate_only_role_play_true(self):  # A+subject+expertise+ estimate
        front_prompt = f"""A is an {self.expertise} in {self.subject}. Read the question, and after considering A’s ability, estimate how many questions A can answer correctly (choose a number from 0-10). Do not provide any explanation.
Questions:\n"""
        back_prompt = "Estimate how many questions you think A can answer correctly: <choose a number from 0-10>.\n"
        return self._create_quiz_prompts(front_prompt, back_prompt)

    def prompt_create_quiz_expertise_estimate_only_random_person(self):  # A random person+ estimate
        front_prompt = f"""A is a randomly chosen person. Read the question, consider A’s ability, estimate how many questions A can answer correctly (choose a number from 0-10). Do not provide any explanation.
Questions:\n
"""
        back_prompt = "Estimate how many questions you think A can answer correctly: <choose a number from 0-10>.\n"
        return self._create_quiz_prompts(front_prompt, back_prompt)

    def create_prompts(self):
        if self.experiment_type == 'answer_only':
            if self.expertise is None or self.subject is None:
                return self.prompt_create_quiz_expertise_answer_only_random_person()
            else:
                return self.prompt_create_quiz_expertise_answer_only_role_play_true()
        elif self.experiment_type == 'estimate_only':
            if self.expertise is None or self.subject is None:
                return self.prompt_create_quiz_expertise_estimate_only_random_person()
            else:
                return self.prompt_create_quiz_expertise_estimate_only_role_play_true()
        else:
            raise ValueError(f"Unknown experiment type: {self.experiment_type}")

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
                answers.append(answer if answer in ['A', 'B', 'C', 'D'] else 'Error')  # Default to 'C' if invalid answer

        return answers

    def extract_estimates_only(self, response):
        estimate_match = re.search(r'(\d+)', response)
        print('extract estimate_match:', estimate_match)
        return int(estimate_match.group(1)) if estimate_match else 0  # Default to 0 if no estimate is found


    def process_dataset(self):
        print(f'Processing dataset for {self.experiment_type}...')
        for complete_prompt in self.prompt_quiz_list:
            # print(complete_prompt)
            response = completion(complete_prompt, self.model)
            print(response)

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
        num_groups = num_questions // 10

        # Ensure all lists are the same length
        llm_answers_filled = (self.llm_answers + ['N/A'] * num_questions)[:num_questions]
        correct_answers_filled = (self.correct_answers_list + ['N/A'] * num_questions)[:num_questions]
        questions = self.dataset['question'][:num_questions].tolist()

        # Pack every 10 answers into a list
        def pack_into_lists(data, size=10):
            return [data[i:i + size] for i in range(0, len(data), size)]

        packed_llm_answers = pack_into_lists(llm_answers_filled) if self.experiment_type != 'estimate_only' else [[ 'N/A'] * 10] * num_groups
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
            llm_confidences = [i / 10 if i is not None else None for i in estimates_filled]

        data = {
            'quiz_id': [f'Quiz_{i + 1}' for i in range(num_groups)],
            'question': packed_questions,
            'llm_answers': packed_llm_answers,
            'correct_answers': packed_correct_answers,
            'estimates': estimates_filled,
            'llm_confidences': llm_confidences,
            'quiz_accuracy': quiz_accuracies,
            'parameters': self.experiment_type,
            'expertise': self.expertise,
            'subject': self.subject,
            'model_name': self.model
        }

        return pd.DataFrame(data)




import warnings
# Suppress only FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
def combine_dataframes(answer_only, estimate_only):
    # Create a copy of the answer_only DataFrame
    combined_df = answer_only.copy()

    # Fill missing values in answer_only from estimate_only where applicable
    combined_df['estimates'].fillna(estimate_only['estimates'], inplace=True)
    combined_df['llm_confidences'].fillna(estimate_only['llm_confidences'], inplace=True)

    # Fill missing values in estimate_only from answer_only
    combined_df['llm_answers'].fillna(answer_only['llm_answers'], inplace=True)
    combined_df['quiz_accuracy'].fillna(answer_only['quiz_accuracy'], inplace=True)

    # Add a new 'parameters' column with the value 'combined'
    combined_df['parameters'] = 'combined'

    # Return the combined DataFrame
    return combined_df
def run_quizlike_pipeline(dataset_name, dataset, output_dir, model, expertise, subject):
    # === Run Answer-Only ===
    processor_answer = OverplacementProcessor(
        dataset,
        experiment_type='answer_only',
        expertise=expertise,
        subject=subject,
        model=model
    )
    df_answer = processor_answer.dataframe
    df_answer.to_csv(os.path.join(output_dir, f"{dataset_name}_answer_only.csv"), index=False)

    # === Run Estimate-Only ===
    processor_estimate = OverplacementProcessor(
        dataset,
        experiment_type='estimate_only',
        expertise=expertise,
        subject=subject,
        model=model
    )
    df_estimate = processor_estimate.dataframe
    df_estimate.to_csv(os.path.join(output_dir, f"{dataset_name}_estimate_only.csv"), index=False)

    # === Combine ===
    combined_df = combine_dataframes(df_answer, df_estimate)
    combined_df.to_csv(os.path.join(output_dir, f"{dataset_name}_combined.csv"), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs='+', required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4")
    parser.add_argument("--expertise", type=str, default=None)
    parser.add_argument("--subject", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_map = load_datasets()

    for dataset_name in args.datasets:
        if dataset_name not in dataset_map:
            print(f"Dataset {dataset_name} not found in dataset_map. Skipping.")
            continue
        print(f"Running overplacement experiment on {dataset_name}...")
        run_quizlike_pipeline(
            dataset_name,
            dataset_map[dataset_name],
            args.output_dir,
            args.model,
            args.expertise,
            args.subject
        )

if __name__ == "__main__":
    main()