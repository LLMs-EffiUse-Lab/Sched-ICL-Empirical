
from datasets import load_dataset
import pandas as pd
import re



def get_query_for_dataset_(system, contents, type):
    def load_train_data(dataset, shot):
        dataset = load_dataset('json', data_files=f"dataset/raw_data/{dataset}/{shot}shot/1.json")
        examples = [(x['text'], x['label']) for x in dataset['train']]
        return examples

    def load_test_data(dataset):
        logs = pd.read_csv(f"dataset/raw_data/{dataset}/{dataset}_2k.log_structured_corrected.csv")
        return logs.Content.tolist()

    def get_log_messages(dataset, shot):
        train, test = [], []
        if shot > 0:
            demos = load_train_data(dataset, shot)
            for demo in demos:
                train.append((demo[0].strip(), demo[1].strip()))
        test_logs = load_test_data(dataset)
        for i, log in enumerate(test_logs):
            test.append(log.strip())
        return train, test

    def generate_formatted_prompt(demos, content):
        fewshot_prompt = '''You will be provided with a log message. You must identify and abstract all the dynamic variables in logs with {{placeholders}} and output a static log template. For example:
    '''
        for log, template in demos:
            # Replace placeholders in the template with {{placeholder}}
            template = re.sub(r'{(\w+)}', r'{{\1}}', template)
            fewshot_prompt += f"The template of '{log}' is '{template}'.\\n\\n"

        fewshot_prompt += '''The template of '{input}' is ' '''

        # Escape special characters in content
        escaped_content = re.escape(content)
        # Replace {input} placeholder with escaped content
        # formatted_prompt = re.sub(r'\{input\}', escaped_content, fewshot_prompt)
        formatted_prompt = re.sub(r'\{input\}', content, fewshot_prompt)

        return formatted_prompt

    # def generate_formatted_prompt(demos, content):
    #     fewshot_prompt = '''You will be provided with a log message. You must identify and abstract all the dynamic variables in logs with {{placeholders}} and output a static log template.
    # For example:
    # '''
    #
    #     for i, (log, template) in enumerate(demos, start=1):
    #         # Check if the template contains complex structures
    #         if re.search(r'{.*=.*}', template):
    #             # Handle complex structures separately
    #             template = re.sub(r'{(\w+)=([^}]+)}', r'{\1={{\2}}}', template)
    #             template = re.sub(r'{(\w+)}', r'{{\1}}', template)
    #         else:
    #             # Replace placeholders in the template with {{placeholder}}
    #             template = re.sub(r'{(\w+)}', r'{{\1}}', template)
    #
    #         fewshot_prompt += f"The template of '{log}' is '{template}'.\n\n"
    #
    #     fewshot_prompt += '''The template of '{input}' is '
    # '''
    #
    #     formatted_prompt = fewshot_prompt.format(input=content)
    #     return formatted_prompt


    simple_prompt = """You will be provided with a log message delimited by backticks. Please extract the log template from this log message: .
                    """
    standard_prompt = """You will be provided with a log message delimited by backticks. You must abstract variables with {{placeholders}} to extract the corresponding template. Print the input log’s template delimited by backticks. Log message:: 
                    """
    enhance_prompt = """You will be provided with a log message delimited by backticks. You must identify and abstract all the dynamic variables in logs with {{placeholders}} and output a static log template. Print the input log’s template delimited by backticks. Log message:. 
                    """

    formatted_prompts = []
    if type == "simple":
        for content in contents:
            formatted_prompts.append(simple_prompt.format(input=content))
    elif type == "standard":
        for content in contents:
            formatted_prompts.append(standard_prompt.format(input=content))
    elif type == "enhance":
        for content in contents:
            formatted_prompts.append(enhance_prompt.format(input=content))
    elif type == "fewshot_1":
        demos = load_train_data(system, 1)
        for content in contents:
            formatted_prompts.append(generate_formatted_prompt(demos, content))
    elif type == "fewshot_2":
        demos = load_train_data(system, 2)
        for content in contents:
            formatted_prompts.append(generate_formatted_prompt(demos, content))
    elif type == "fewshot_4":
        demos = load_train_data(system, 4)
        for content in contents:
            formatted_prompts.append(generate_formatted_prompt(demos, content))

    return formatted_prompts


def get_whole_query_(system, content, type):
    def load_train_data(dataset, shot):
        dataset = load_dataset('json', data_files=f"dataset/raw_data/{dataset}/{shot}shot/1.json")
        examples = [(x['text'], x['label']) for x in dataset['train']]
        return examples

    def load_test_data(dataset):
        logs = pd.read_csv(f"dataset/raw_data/{dataset}/{dataset}_2k.log_structured_corrected.csv")
        return logs.Content.tolist()

    def get_log_messages(dataset, shot):
        train, test = [], []
        if shot > 0:
            demos = load_train_data(dataset, shot)
            for demo in demos:
                train.append((demo[0].strip(), demo[1].strip()))
        test_logs = load_test_data(dataset)
        for i, log in enumerate(test_logs):
            test.append(log.strip())
        return train, test

    def generate_formatted_prompt(demos, content):
        fewshot_prompt = '''You will be provided with a log message. You must identify and abstract all the dynamic variables in logs with {{placeholders}} and output a static log template. 
    For example:
    '''

        for i, (log, template) in enumerate(demos[0], start=1):
            # Replace placeholders in the template with {{placeholder}}
            template = re.sub(r'{(\w+)}', r'{{\1}}', template)
            fewshot_prompt += f"The template of '{log}' is '{template}'.\n\n"

        fewshot_prompt += '''The template of '{input}' is '
    '''

        formatted_prompt = fewshot_prompt.format(input=content)
        return formatted_prompt

    simple_prompt = """You will be provided with a log message delimited by backticks. Please extract the log template from this log message: .
                    """
    standard_prompt = """You will be provided with a log message delimited by backticks. You must abstract variables with {{placeholders}} to extract the corresponding template. Print the input log’s template delimited by backticks. Log message:: 
                    """
    enhance_prompt = """You will be provided with a log message delimited by backticks. You must identify and abstract all the dynamic variables in logs with {{placeholders}} and output a static log template. Print the input log’s template delimited by backticks. Log message:. 
                    """

    if type == "simple":
        formatted_prompt = simple_prompt.format(input=content)
    elif type == "standard":
        formatted_prompt = standard_prompt.format(input=content)
    elif type == "enhance":
        formatted_prompt = enhance_prompt.format(input=content)
    elif type == "fewshot_1":
        demos = load_train_data(system, 1)
        formatted_prompt = generate_formatted_prompt(demos, content)
    elif type == "fewshot_2":
        demos = load_train_data(system, 2)
        formatted_prompt = generate_formatted_prompt(demos, content)
    elif type == "fewshot_4":
        demos = load_train_data(system, 4)
        formatted_prompt = generate_formatted_prompt(demos, content)

    return formatted_prompt