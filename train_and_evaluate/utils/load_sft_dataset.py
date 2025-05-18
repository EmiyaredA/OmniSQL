import json
import torch
from torch.utils.data import Dataset

def find_sublist_index(lst, sublist):
    sublist_length = len(sublist)
    for i in range(len(lst) - sublist_length + 1):
        if lst[i:i + sublist_length] == sublist:
            return i
    return -1

def obtain_labels(input_ids, assistant_start_token_ids):
    assistant_start_idx = find_sublist_index(input_ids, assistant_start_token_ids)
    if assistant_start_idx == -1:
        labels = input_ids
        print("length of the output sequence exceeds max length")
    else:
        labels = [-100] * assistant_start_idx + input_ids[assistant_start_idx:]
    assert len(input_ids) == len(labels)
    return labels

class SFTDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length, mode):
        super().__init__()
        self.mode = mode
        self.data_path = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.assistant_start_token_ids = [151644, 77091]  # for Qwen2.5's tokenizer

        if not data_dir.endswith(".jsonl"):
            raise ValueError(
                f"[ERROR] {mode} 模式只支持 .jsonl 格式：{data_dir}\n"
                f"请先将数据转换为每行一个 JSON 对象的 .jsonl 文件。"
            )

        # 预处理：记录每行的起始字节偏移位置（用于 seek）
        self.line_offsets = []
        with open(self.data_path, "rb") as f:
            offset = f.tell()
            line = f.readline()
            while line:
                self.line_offsets.append(offset)
                offset = f.tell()
                line = f.readline()

        print(f"[INFO] 载入 {len(self.line_offsets)} 条样本 ({mode})")

    def __len__(self):
        return len(self.line_offsets)

    def __getitem__(self, index):
        with open(self.data_path, "r", encoding="utf-8") as f:
            f.seek(self.line_offsets[index])
            data = json.loads(f.readline())

        if self.mode == "pre-train":
            return {
                "input_ids": torch.tensor(data["input_ids"], dtype=torch.int64),
                "attention_mask": torch.tensor(data["attention_mask"], dtype=torch.int64),
                "labels": torch.tensor(data["labels"], dtype=torch.int64)
            }

        elif self.mode == "sft":
            prompt = self.tokenizer.apply_chat_template([
                {"role": "user", "content": data["input_seq"]},
                {"role": "assistant", "content": data["output_seq"]}
            ], add_generation_prompt=False, tokenize=False)

            input_ids = self.tokenizer(prompt, truncation=False)["input_ids"]

            if len(input_ids) > self.max_length:
                input_ids = input_ids[-self.max_length:]

            attention_mask = [1] * len(input_ids) + [0] * (self.max_length - len(input_ids))
            labels = obtain_labels(input_ids, self.assistant_start_token_ids) + [-100] * (self.max_length - len(input_ids))
            input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.int64),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.int64),
                "labels": torch.tensor(labels, dtype=torch.int64)
            }
