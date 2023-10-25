from datasets import load_dataset, Dataset, Features, Value, ClassLabel
import pandas as pd
import os
import requests
import jsonlines

path = f'{os.getcwd()}/data'
# path = f'{os.getcwd()}/toy_submission/llama_recipes/data'


def download_file(path_destination):
    categories = [
        "Age",
        "Disability_status",
        "Gender_identity",
        "Nationality",
        "Physical_appearance",
        "Race_ethnicity",
        "Race_x_SES",  # extra intersectional category as mentioned in section 3.2
        "Race_x_gender",  # extra intersectional category as mentioned in section 3.2
        "Religion",
        "SES",
        "Sexual_orientation",
    ]

    for category in categories:
        response = requests.get(
            f'https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/{category}.jsonl')
        print(response.raise_for_status())

        with open(f'{path_destination}/data.jsonl', "wb") as file:
            file.write(response.content)


class TsotsaDataset:
    def __init__(self, split, type_dataset="bb", name="lima"):
        self.dataset = []
        self.dataset_id = ""
        self.split = split
        self.type_dataset = type_dataset
        self.dataset_name = name

    # get size of the dataset

    def __len__(self):
        return len(self.dataset)

    """
        getter methods
    """

    def get_name(self):
        return self.dataset_name

    def get_type(self):
        return self.type_dataset

    def get_dataset(self):
        return self.dataset

    """_
        BIG Bench Scenario
    """
    # load lima dataset

    def _load_lima(self):
        self.dataset_id = "GAIR/lima"
        self.dataset = load_dataset(self.dataset_id, split=self.split)
        return self.dataset

    # load databricks dataset
    def _load_dolly(self):
        self.dataset_id = "databricks/databricks-dolly-15k"
        self.dataset = load_dataset(self.dataset_id, split=self.split)
        return self.dataset

    # load oasst1 dataset

    def _load_oasst1(self):
        self.dataset_id = "OpenAssistant/oasst1"
        self.dataset = load_dataset(self.dataset_id, split=self.split)
        return self.dataset

    # summerization dataset
    def _load_redpajama(self):
        self.dataset_id = "togethercomputer/RedPajama-Data-1T"
        self.dataset = load_dataset(
            self.dataset_id, split=self.split, name="arxiv")
        return self.dataset

    """_
        truthfullqa Scenario
    """
    # truthfullqa dataset

    def _load_ai2_arc(self):
        self.dataset_id = "ai2_arc"
        self.dataset = load_dataset(
            self.dataset_id, split=self.split, name="ARC-Easy")

        return self.dataset

    # truthfullqa dataset
    def _load_commonsense_qa(self):
        self.dataset_id = "commonsense_qa"
        self.dataset = load_dataset(self.dataset_id, split=self.split)
        return self.dataset

    def _load_truthfulqa(self):
        self.dataset_id = "truthful_qa"
        self.dataset = load_dataset(
            self.dataset_id, split='validation', name='generation')
        return self.dataset

    def _load_truthfulqa1(self):
        self.dataset_id = "truthful_qa"
        self.dataset = load_dataset(
            self.dataset_id, split='validation', name='multiple_choice')
        return self.dataset

    """ 
        Summarization 
    """

    def _load_xsum(self):
        self.dataset_id = "xsum"
        self.dataset = load_dataset(self.dataset_id, split=self.split)
        return self.dataset

    def _load_cnn_dailymail(self):
        self.dataset_id = "cnn_dailymail"
        self.dataset = load_dataset(
            self.dataset_id, split=self.split, name='3.0.0')
        return self.dataset

    """ 
        BBQ Datasete
    """

    def _load_bbq(self):
        # download_file(path)
        for file in os.listdir(path):
            print(file)
            if file.endswith(".jsonl"):
                with jsonlines.open(f'{path}/{file}') as reader:
                    for data in reader:
                        self.dataset.append(data)

        self.dataset = self.dataset[:10000]
        # print(self.dataset.columns)
        print("Size of dataset", len(self.dataset))
        return self.dataset

    """ 
        formating function for each type of scenarios
    """

    def prepare_bb_scenario(self, sample):

        if 'conversations' in sample:
            instruction = sample['conversations'][0]
            response = sample['conversations'][1]
            del sample['conversations']
            sample['instruction'] = instruction
            sample['response'] = response

        # if 'role' in sample:
        #     conversations = sample['text']
        #     for i in range(len(conversations)):
        #         if sample['role'] == 'assistant':
        #             sample['response'] = conversations[i]
        #         else:
        #             sample['instruction'] = conversations[i]
        string = f"""
        ### Welcome in your assistant!!!!!!!
                
        ### INSTRUCTIONS:
        \n{sample['instruction']}
                
        ### ANSWER:
        \n{sample['response']}
        """
        return string

    def prepare_truthfulqa_scenario(self, sample):

        formatted_list = []
        if 'best_answer' in sample:
            choices = sample['correct_answers']
            best_answer = sample['best_answer']
            sample['answerKey'] = best_answer
            del sample['best_answer']
            label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
            size_choices = len(choices)
            reference = {
                'label': label[:size_choices],
                'text': choices
            }
            for i in range(len(reference['label'])):
                if best_answer == choices[i]:
                    sample['answerKey'] = f"{reference['label'][i]}. {sample['answerKey']}"
                formatted_list.append(
                    f"{reference['label'][i]}. {reference['text'][i]} \n\t")

        elif 'mc2_targets' in sample:
            sample_dict = sample['mc2_targets']
            choices = sample_dict['choices']
            label_list = sample_dict['labels']
            label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
            size_choices = len(choices)
            sample['answerKey'] = []

            reference = {
                'label': label[:size_choices],
                'text': choices
            }

            for i in range(len(reference['label'])):
                if label_list[i] == 1:
                    sample['answerKey'].append(f"{reference['label'][i]}")

                formatted_list.append(
                    f"{reference['label'][i]}. {reference['text'][i]} \n\t")
            sample['answerKey'] = ', '.join(sample['answerKey'])
        else:
            sample_dict = sample['choices']
            text_list = sample_dict['text']
            label_list = sample_dict['label']
            for i in range(len(text_list)):
                formatted_list.append(f"{label_list[i]}. {text_list[i]} \n\t")
        string = f"""
        ### Welcome in your assistant!!!!!!!
            
        ### INSTRUCTIONS:
        given a question and multi choices options, you will get the correct answer\n
        Question: {sample['question']}
            
        {"".join(formatted_list)}

        ### Answer
        {sample['answerKey']}

        """
        # print(string)
        return string

    def prepare_summerization_scenario(self, sample):
        if 'highlights' and 'article' in sample:
            summary = sample['highlights']
            document = sample['article']
            del sample['highlights']
            del sample['article']
            sample['summary'] = summary
            sample['document'] = document
        string = f"""
            ### Welcome in your assistant!!!!!!!
            
            ### INSTRUCTIONS:
            please give an input a document or article to recive the his summary
            {sample['document']}


            ### Summary
            {sample['summary']}

        """
        # print(string)
        return string

    def prepare_bbq_scenario(self, sample):

        ans0 = sample['ans0']
        ans1 = sample['ans1']
        ans2 = sample['ans2']
        label = sample['label']
        question = sample['question']
        context = sample['context']
        context_condition = ''
        question_polarity = ''
        if sample['question_polarity'] == 'nonneg':
            question_polarity = 'Postitve'
        else:
            question_polarity = 'Negative'

        if sample['context_condition'] == 'ambig':
            context_condition = 'Ambigous'
        else:
            context_condition = 'Non-Ambigous'
        category = sample['category']

        choices = []
        reference = {
            'label': ['A', 'B', 'C'],
            'text': [ans0, ans1, ans2]
        }

        correct_answer = reference['label'][label]
        for i in range(len(reference['label'])):
            choices.append(
                f'{reference["label"][i]}. {reference["text"][i]} \n\t')
        string = f"""
        ### Welcome in your assistant!!!!!!!
            
        ### Context
        {context}
        {question_polarity}
        {context_condition}   
            
        ### INSTRUCTIONS:  
        {question}
        
        {''.join(choices)}
            
        #### Category
        {category}

        ### Answer
        {correct_answer}

        """
        # print(string)
        return string
