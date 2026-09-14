import os, json
import random

# Project root (load_dataset.py lives under utils/, one level up)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATASET_DIR = os.path.join(_ROOT, "datasets")

# Dataset keyword -> relative path under datasets/
_DATASET_MAP = {
    "gaokao_math_ii":  "GAOKAO-BENCH/data/Multiple-choice_Questions/2010-2022_Math_II_MCQs.json",
    "gaokao_math_i":   "GAOKAO-BENCH/data/Multiple-choice_Questions/2010-2022_Math_I_MCQs.json",
    "gaokao_history":  "GAOKAO-BENCH/data/Multiple-choice_Questions/2010-2022_History_MCQs.json",
    "gaokao_biology":  "GAOKAO-BENCH/data/Multiple-choice_Questions/2010-2022_Biology_MCQs.json",
    "wic":             "SuperGLUE/WiC/val.jsonl",
    "triviaqa":        "triviaqa/triviaqa-train.jsonl",
    "race_middle":     "race/validation/middle.jsonl",
    "race_high":       "race/validation/high.jsonl",
    "gsm8k":           "gsm8k/train.jsonl",
}


def _ensure_path(keyword: str) -> str:
    """
    Find the local datasets/ path for the given keyword.
    If the file does not exist, call download.py to auto-download it.
    """
    kw = keyword.lower()
    # Resolve named datasets exactly first: gaokao_math_i is a prefix of
    # gaokao_math_ii, so substring matching alone selects the wrong file.
    if kw in _DATASET_MAP:
        full = os.path.join(_DATASET_DIR, _DATASET_MAP[kw])
        if not os.path.isfile(full):
            from download import ensure_dataset
            full = ensure_dataset(kw)
        return full
    # Match keyword against _DATASET_MAP keys and relative paths
    for name, rel in _DATASET_MAP.items():
        if kw in name or kw in rel.lower():
            full = os.path.join(_DATASET_DIR, rel)
            if not os.path.isfile(full):
                from download import ensure_dataset
                full = ensure_dataset(kw)
            return full
    # If keyword is already a valid file path, use it directly
    if os.path.isfile(keyword):
        return keyword
    # Fall back to download.py
    from download import ensure_dataset
    return ensure_dataset(kw)


def get_path():
    """Return a list of local paths for all known datasets (used by load_prefetch_random)."""
    return [os.path.join(_DATASET_DIR, rel) for rel in _DATASET_MAP.values()]


def load_prefetch_random(dataset_path='GAOKAO', batch_size=1, input_num=50, idx=0, range=1):
    path_set = get_path()
    all_inputs = []
    for path in path_set:
        if dataset_path.lower() in path.lower() and os.path.isfile(path):
            one_inputs = load_GAOKAO_MCQs(path, batch_size, input_num)
            all_inputs.extend(one_inputs)
    random.shuffle(all_inputs)
    return all_inputs


def load_all(dataset_path, batch_size=1, input_num=100, idx=0, range=1):
    # If not an existing file path, resolve via keyword (with auto-download)
    if not os.path.isfile(dataset_path):
        dataset_path = _ensure_path(dataset_path)

    print('dataset_path:', dataset_path)

    if "2010-2022_Math_II_MCQs.json" in dataset_path:
        return load_GAOKAO_MCQs(dataset_path, batch_size, input_num)
    elif "2010-2022_Math_I_MCQs.json" in dataset_path:
        return load_GAOKAO_MCQs(dataset_path, batch_size, input_num)
    elif "2010-2022_History_MCQs.json" in dataset_path:
        return load_GAOKAO_MCQs(dataset_path, batch_size, input_num)
    elif "2010-2022_Biology_MCQs.json" in dataset_path:
        return load_GAOKAO_MCQs(dataset_path, batch_size, input_num)
    elif "SuperGLUE/WiC" in dataset_path or "wic" in dataset_path.lower():
        return load_superglue_wic(dataset_path, batch_size, input_num)
    elif "triviaqa" in dataset_path.lower():
        return load_triviaqa(dataset_path, batch_size, input_num)
    elif "middle" in dataset_path:
        return load_race(dataset_path, batch_size, input_num)
    elif "high" in dataset_path:
        return load_race(dataset_path, batch_size, input_num)
    elif "gsm8k" in dataset_path.lower():
        return load_gsm8k(dataset_path, batch_size, input_num)
    else:
        raise ValueError(f"Unrecognized dataset path: {dataset_path}")


def load_superglue_wic(dataset_path, batch_size=1, input_num=100, idx=0, ranges=1):
    prompt_template = ("Sentence 1: {sentence1}\nSentence 2: {sentence2}\n"
                       "Are '{word}' in the above two sentenses the same?\nA. Yes\nB. No\nAnswer:")
    questions = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        line_count = 0
        for line in f:
            try:
                data = json.loads(line.strip())
                prompt = prompt_template.format(
                    word=data['word'],
                    sentence1=data['sentence1'].strip(),
                    sentence2=data['sentence2'].strip()
                )
                questions.append(prompt)
                line_count += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping line {line_count} (parse error): {str(e)}")
    all_inputs = []
    for i in range(0, len(questions), batch_size):
        all_inputs.append(questions[i:i+batch_size])
    all_inputs = all_inputs[:input_num]
    return [batch for batch in all_inputs if len(batch) > 0]


def load_GAOKAO_MCQs(dataset_path, batch_size=1, input_num=100, idx=0, ranges=1):
    if "2010-2022_Math_II_MCQs.json" in dataset_path:
        subject = 'Math'
    elif "2010-2022_Math_I_MCQs.json" in dataset_path:
        subject = 'Math'
    elif "2010-2022_History_MCQs.json" in dataset_path:
        subject = 'History'
    elif "2010-2022_Biology_MCQs.json" in dataset_path:
        subject = 'Biology'
    else:
        subject = 'Unknown'
    prefix_template = (
        f"Please answer the following {subject} multiple-choice question.\n"
        f"Think step by step and write your reasoning between [Analysis] and <eoe>.\n"
        f"Choose the correct answer from A, B, C, D and write it between [Answer] and <eoa>.\n"
        f"Example: [Answer]: A <eoa>\n"
        f"Full answer format:\n[Analysis] ... <eoe>\n[Answer] ... <eoa>\n"
        f"Strictly follow the format above.\nQuestion:\n"
    )
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        question_list = data["example"]
    questions = [prefix_template + item['question'] for item in question_list]
    all_inputs = []
    for i in range(0, len(questions), batch_size):
        all_inputs.append(questions[i:i+batch_size])
    all_inputs = all_inputs[:input_num]
    return all_inputs


def load_triviaqa(dataset_path, batch_size=1, input_num=100, idx=0, ranges=1):
    prompt_template = ("Answer these questions, your answer should be as simple as possible, "
                       "start your answer with the prompt 'The answer is '.\nQ: {question}")
    questions = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        line_count = 0
        for line in f:
            try:
                data = json.loads(line.strip())
                prompt = prompt_template.format(question=data['question'].strip())
                questions.append(prompt)
                line_count += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping line {line_count} (parse error): {str(e)}")
    all_inputs = []
    for i in range(0, len(questions), batch_size):
        all_inputs.append(questions[i:i+batch_size])
    all_inputs = all_inputs[:input_num]
    return [batch for batch in all_inputs if len(batch) > 0]


def load_race(dataset_path, batch_size=1, input_num=100, idx=0, ranges=1):
    prompt_template = ("Read the article, and answer the question by replying A, B, C or D.\n\n"
                       "Article:\n{article}\n\nQ: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}")
    questions = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        line_count = 0
        for line in f:
            try:
                data = json.loads(line.strip())
                prompt = prompt_template.format(
                    article=data['article'],
                    question=data['question'].strip(),
                    A=data['options'][0].strip(),
                    B=data['options'][1].strip(),
                    C=data['options'][2].strip(),
                    D=data['options'][3].strip()
                )
                questions.append(prompt)
                line_count += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping line {line_count} (parse error): {str(e)}")
    all_inputs = []
    for i in range(0, len(questions), batch_size):
        all_inputs.append(questions[i:i+batch_size])
    all_inputs = all_inputs[:input_num]
    return [batch for batch in all_inputs if len(batch) > 0]


def load_gsm8k(dataset_path, batch_size=1, input_num=100, idx=0, ranges=1):
    prompt_template = (
        "Question: Angelo and Melanie want to plan how many hours over the next week they should study together "
        "for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. "
        "They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for "
        "each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to "
        "study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack "
        "breaks each day, and 30 minutes for lunch each day?\nLet's think step by step\nAnswer:\n"
        "Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, "
        "3 hours x 2 chapters = 6 hours total.\n"
        "For the worksheets they plan to dedicate 1.5 hours for each worksheet, "
        "1.5 hours x 4 worksheets = 6 hours total.\n"
        "Angelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\n"
        "However, they need to include time for breaks and lunch. Every hour they want to include a 10-minute "
        "break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\n"
        "They also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\n"
        "And they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for "
        "snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\n"
        "So Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\n"
        "They want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\n"
        "They will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4\n\n"
        "Question:{question}.Let's think step by step\nAnswer:"
    )
    questions = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        line_count = 0
        for line in f:
            try:
                data = json.loads(line.strip())
                prompt = prompt_template.format(question=data['question'])
                questions.append(prompt)
                line_count += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping line {line_count} (parse error): {str(e)}")
    all_inputs = []
    for i in range(0, len(questions), batch_size):
        all_inputs.append(questions[i:i+batch_size])
    all_inputs = all_inputs[:input_num]
    return [batch for batch in all_inputs if len(batch) > 0]


def load_gsm8k_simple(dataset_path, batch_size=1, input_num=100, idx=0, ranges=1):
    questions = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                questions.append(entry['question'])
    all_inputs = []
    for i in range(0, len(questions), batch_size):
        all_inputs.append(questions[i:i+batch_size])
    all_inputs = all_inputs[:input_num]
    return all_inputs


import torch

def load_deepseekr1(dataset_path, batch_size=1, input_num=100, idx=0, range=1):
    if batch_size == 0:
        raise ValueError(f"batch size {batch_size} is wrong")
    from torch.utils.data import Dataset, DataLoader
    from datasets import load_from_disk

    dataset = load_from_disk(dataset_path)
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: [item["input"] for item in batch]
    )
    max_batches = input_num
    all_inputs = []
    for batch_idx, batch_data in enumerate(dl):
        if batch_idx >= max_batches:
            break
        all_inputs.append(batch_data)
    return all_inputs


def load_all_pt(dataset_path, batch_size=1):
    print(dataset_path)
    prompts = torch.load(dataset_path)

    from torch.utils.data import Dataset
    class StringListDataset(Dataset):
        def __init__(self, string_list):
            self.string_list = string_list
        def __len__(self):
            return len(self.string_list)
        def __getitem__(self, idx):
            return self.string_list[idx]
    ds = StringListDataset(prompts)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


def load_all_arrow(dataset_path, batch_size=2):
    from torch.utils.data import Dataset, DataLoader
    from datasets import load_from_disk

    print(dataset_path)
    dataset = load_from_disk(dataset_path)
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: [item["question"] for item in batch]
    )


if __name__ == '__main__':
    dataset_path = 'wic'
    all_inputs = load_all(dataset_path, batch_size=1, input_num=5)
    for i, inputs in enumerate(all_inputs):
        print('i=', i, inputs)
