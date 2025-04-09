from utils.llm_api_utils import completion
import pandas as pd
import re
import warnings
import argparse
import os
from utils.dataloader import load_datasets
warnings.simplefilter(action='ignore', category=FutureWarning)


class QuizLikeDatasetProcessor_DemographicClass:
    def __init__(self, dataset, race, age, gender, experiment_type, role_play=False, model='claude-3-sonnet-20240229'):
        self.dataset = dataset
        self.race = race
        self.age = age
        self.gender = gender
        self.experiment_type = experiment_type
        self.role_play = role_play
        self.model = model
        self.correct_answers_list = dataset['answer'].tolist()
        self.max_questions = (len(dataset['question']) // 10) * 10
        self.quizzes = self._get_quizzes()

        self.prompt_quiz_list = self._create_prompts()
        self.llm_answers = []
        self.estimates = []

        self._process_dataset()
        self.dataframe = self._create_dataframe()

    def _get_quizzes(self, questions_per_quiz=10):
        num_questions = min(self.max_questions, len(self.dataset['question']))
        selected = {col: self.dataset[col][:num_questions].tolist()
                    for col in ['question', 'A', 'B', 'C', 'D']}
        return [
            {key: selected[key][i:i + questions_per_quiz] for key in selected}
            for i in range(0, num_questions, questions_per_quiz)
        ]

    def _create_prompts(self):
        if self.experiment_type == 'answer_only':
            front = f"You are a {self.race} {self.gender} in the {self.age} age group. Read the question and provide your answer (only the option letter). Do not provide any explanation.\nQuestions:\n"
            back = "\n".join([f"{i+1}. <Only Option Letter >" for i in range(10)]) + "\n"
        elif self.experiment_type == 'estimate_only':
            front = f"You are a {self.race} {self.gender} in the {self.age} age group. Read the question, estimate how many questions you can answer correctly (choose a number from 0-10). Do not provide any explanation.\nQuestions:\n"
            back = "Estimate how many questions you can answer correctly: <only choose one number from 0-10>.\n"
        else:
            raise ValueError(f"Unknown experiment type: {self.experiment_type}")

        return [front + self._format_questions(quiz) + back for quiz in self.quizzes]

    def _format_questions(self, quiz):
        return "\n".join([
            f"{i+1}. {quiz['question'][i]}\n   A. {quiz['A'][i]}\n   B. {quiz['B'][i]}\n   C. {quiz['C'][i]}\n   D. {quiz['D'][i]}"
            for i in range(10)
        ])

    def _process_dataset(self):
        for prompt in self.prompt_quiz_list:
            response = completion(prompt, self.model)
            if self.experiment_type == 'answer_only':
                answers = self._extract_answers(response)
                self.llm_answers.extend(answers)
            elif self.experiment_type == 'estimate_only':
                estimate = self._extract_estimate(response)
                self.estimates.append(estimate)

    def _extract_answers(self, response):
        lines = response.strip().split("\n")
        answers = []
        for line in lines:
            line = line.strip()
            if line in ['A', 'B', 'C', 'D']:
                answers.append(line)
            else:
                m = re.match(r'^\d+\.\s*([A-D])\b', line)
                answers.append(m.group(1) if m else 'C')
        return answers

    def _extract_estimate(self, response):
        m = re.search(r'\b([0-9]|10)\b', response)
        return int(m.group(1)) if m else None

    def _create_dataframe(self):
        num_groups = self.max_questions // 10
        correct = (self.correct_answers_list + ['N/A'] * self.max_questions)[:self.max_questions]

        def pack(data): return [data[i:i + 10] for i in range(0, len(data), 10)]

        packed_correct = pack(correct)

        if self.experiment_type != 'estimate_only':
            answers = (self.llm_answers + ['N/A'] * self.max_questions)[:self.max_questions]
            packed_answers = pack(answers)
        else:
            packed_answers = [['N/A'] * 10] * num_groups

        if self.experiment_type == 'answer_only':
            estimates = [None] * num_groups
        else:
            estimates = (self.estimates + [None] * num_groups)[:num_groups]

        confidences = [e / 10 if e is not None else None for e in estimates]
        quiz_acc = [
            sum([a == b for a, b in zip(ans, cor)]) / 10
            for ans, cor in zip(packed_answers, packed_correct)
        ]

        return pd.DataFrame({
            'quiz_id': [f'Quiz_{i + 1}' for i in range(num_groups)],
            'question': pack(self.dataset['question'][:self.max_questions].tolist()),
            'gender': self.gender,
            'age': self.age,
            'race': self.race,
            'llm_answers': packed_answers,
            'correct_answers': packed_correct,
            'estimates': estimates,
            'llm_confidences': confidences,
            'quiz_accuracy': quiz_acc,
            'parameters': self.experiment_type,
            'model_name': self.model
        })


def combine_dataframes(answer_only_df, estimate_only_df):
    df = answer_only_df.copy()
    for col in ['estimates', 'llm_confidences']:
        df[col].fillna(estimate_only_df[col], inplace=True)
    df['parameters'] = 'combined'
    return df


def run_quizlike_pipeline_demographic(dataset_name, dataset, output_dir, model, race, age, gender):
    # === Run Answer-Only ===
    processor_answer = QuizLikeDatasetProcessor_DemographicClass(
        dataset=dataset,
        race=race,
        age=age,
        gender=gender,
        experiment_type='answer_only',
        model=model
    )
    df_answer = processor_answer.dataframe
    # df_answer.to_csv(os.path.join(output_dir, f"{dataset_name}_answer_only.csv"), index=False)

    # === Run Estimate-Only ===
    processor_estimate = QuizLikeDatasetProcessor_DemographicClass(
        dataset=dataset,
        race=race,
        age=age,
        gender=gender,
        experiment_type='estimate_only',
        model=model
    )
    df_estimate = processor_estimate.dataframe
    # df_estimate.to_csv(os.path.join(output_dir, f"{dataset_name}_estimate_only.csv"), index=False)

    # === Combine ===
    combined_df = combine_dataframes(df_answer, df_estimate)
    combined_df.to_csv(os.path.join(output_dir, f"{dataset_name}_demographic.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs='+', required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4")
    parser.add_argument("--race", type=str, required=True)
    parser.add_argument("--age", type=str, required=True)
    parser.add_argument("--gender", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_map = load_datasets()

    for dataset_name in args.datasets:
        if dataset_name not in dataset_map:
            print(f"Dataset {dataset_name} not found in dataset_map. Skipping.")
            continue
        print(f"Running demographic quiz-like experiment on {dataset_name}...")
        run_quizlike_pipeline_demographic(
            dataset_name,
            dataset_map[dataset_name],
            args.output_dir,
            args.model,
            args.race,
            args.age,
            args.gender
        )

if __name__ == "__main__":
    main()