from utils.llm_api_utils import completion
from utils.dataloader import load_datasets
import pandas as pd
import re, os, argparse
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
        self.prompt_quiz_list = self.create_prompts()
        self.llm_answers = []
        self.estimates = []

        self.process_dataset()
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
        front_prompt = f"Please answer the following {str(self.questions_per_quiz)} questions by selecting only the option letter. Do not provide any explanation.\nQuestions:\n"
        back_prompt = (
                "Now, please answer these questions following the format:\n" +
                "\n".join([f"{i}. <Only Option Letter>" for i in range(1, self.questions_per_quiz + 1)]) +
                f"\n\nEstimate how many questions you answered correctly: <choose a number from 0-{str(self.questions_per_quiz)}>.\n"
        )
        return self._create_quiz_prompts(front_prompt, back_prompt)

    def _create_quiz_prompts(self, front_prompt, back_prompt):
        quizzes = []
        for quiz in self.quizzes:
            questions = []
            for i in range(len(quiz['question'])):
                question = quiz['question'][i]
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

    def extract_answers_and_estimate(self, response):
        lines = response.strip().split('\n')
        answers = []
        estimate = None

        for line in lines:
            single_letter_match = re.match(r'^[A-D]$', line.strip())
            numbered_match = re.match(r'^\d+\.?\s*([A-D])$', line.strip())

            if single_letter_match:
                answers.append(single_letter_match.group())
            elif numbered_match:
                answers.append(numbered_match.group(1))

        answers = answers[:self.questions_per_quiz]
        while len(answers) < self.questions_per_quiz:
            answers.append('N/A')

        estimate_line = lines[-1]
        estimate_match = re.search(r'(\d+)', estimate_line)
        if estimate_match:
            estimate = int(estimate_match.group(1))

        return answers, estimate

    def process_dataset(self):
        print(f'Processing dataset for {self.experiment_type}...')
        for complete_prompt in self.prompt_quiz_list:
            print(complete_prompt)
            response = completion(complete_prompt, self.model)
            answers, estimate = self.extract_answers_and_estimate(response)
            self.llm_answers.extend(answers)
            self.estimates.append(estimate)

    def calculate_quiz_accuracy(self, llm_answers, correct_answers):
        correct_count = sum([1 for llm, correct in zip(llm_answers, correct_answers) if llm == correct])
        return correct_count / len(correct_answers)

    def create_dataframe(self):
        num_questions = self.max_questions
        num_groups = num_questions // self.questions_per_quiz

        llm_answers_filled = (self.llm_answers + ['N/A'] * num_questions)[:num_questions]
        correct_answers_filled = (self.correct_answers_list + ['N/A'] * num_questions)[:num_questions]
        questions = self.dataset['question'][:num_questions].tolist()

        def pack_into_lists(data, size=self.questions_per_quiz):
            return [data[i:i + size] for i in range(0, len(data), size)]

        packed_llm_answers = pack_into_lists(llm_answers_filled)
        packed_correct_answers = pack_into_lists(correct_answers_filled)
        packed_questions = pack_into_lists(questions)

        quiz_accuracies = [self.calculate_quiz_accuracy(llm_ans, correct_ans)
                           for llm_ans, correct_ans in zip(packed_llm_answers, packed_correct_answers)]

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
            'parameters': 'quiz_like',
            'model_name': self.model
        }

        return pd.DataFrame(data)


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


def main():
    args = parse_arguments()
    dataset_map = load_datasets()
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    if 'all' in args.datasets:
        selected_datasets = list(dataset_map.items())
    else:
        selected_datasets = [(name, df) for name, df in dataset_map.items()
                             if name in args.datasets]

    if not selected_datasets:
        raise ValueError("No valid datasets selected. Check your dataset names.")

    for ds_name, dataset in tqdm(selected_datasets, desc="Processing datasets"):
        try:
            sanitized_name = ds_name.replace(" ", "_")

            processor = QuizLikeDataProcessor(
                dataset=dataset,
                experiment_type='answer_and_estimate',
                questions_per_quiz=args.questions_per_quiz,
                model=args.model,
            )

            filename = f"{sanitized_name}_q{args.questions_per_quiz}_{timestamp}.csv"
            output_path = os.path.join(args.output_dir, filename)
            processor.dataframe.to_csv(output_path, index=False)
            tqdm.write(f"✓ Successfully saved: {output_path}")

        except Exception as e:
            tqdm.write(f"✕ Processing failed [{ds_name}]: {str(e)}")
            continue


if __name__ == "__main__":
    main()