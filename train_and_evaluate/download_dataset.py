from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="seeklhy/OmniSQL-datasets",
#     repo_type="dataset",
#     local_dir="/mnt/cache/tonghao/slns/dbProject2/OmniSQL/train_and_evaluate"
# )
# export HF_ENDPOINT=https://hf-mirror.com

import ijson
import json

def read_large_json_items(path, max_items=None):
    """
    使用 ijson 流式读取大 JSON 文件中的每个 item。
    要求 JSON 文件结构为： [ {...}, {...}, ... ]
    """
    with open(path, 'rb') as f:
        for idx, item in enumerate(ijson.items(f, 'item')):
            print(f"\n[{idx}] JSON 对象内容:")
            print(json.dumps(item, indent=2, ensure_ascii=False))

            if max_items is not None and idx + 1 >= max_items:
                print(f"\n已读取 {max_items} 条记录，提前终止。")
                break
            
def convert_large_json_to_jsonl(input_path, output_path):
    """
    将大 JSON 数组文件（[{}, {}, ...]）转换为 JSONL，每行一个对象，内存友好。
    """
    with open(input_path, 'rb') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
        for i, item in enumerate(ijson.items(f_in, 'item')):
            json_line = json.dumps(item, ensure_ascii=False)
            f_out.write(json_line + '\n')
            if i % 10000 == 0:
                print(f"[INFO] 已处理 {i} 条记录...")

    print(f"\n✅ 转换完成！输出文件: {output_path}")

def count_json_records(file_path):
    count = 0
    with open(file_path, 'rb') as f:
        # 假设 JSON 顶层是数组结构：[{...}, {...}, ...]
        for _ in ijson.items(f, 'item'):
            count += 1
    return count

# 示例调用
if __name__ == "__main__":
    # json_path = "/mnt/cache/tonghao/slns/dbProject2/OmniSQL/train_and_evaluate/data/train_all.json"  # 替换为你的文件路径
    # read_large_json_items(json_path, max_items=1)  # 仅读取前 3 条做示范
    
    # input_file = "/mnt/cache/tonghao/slns/dbProject2/OmniSQL/train_and_evaluate/data/train_all.json"
    # output_file = "/mnt/cache/tonghao/slns/dbProject2/OmniSQL/train_and_evaluate/data/train_all.jsonl"
    # convert_large_json_to_jsonl(input_file, output_file)
    json_file = '/mnt/cache/tonghao/slns/dbProject2/OmniSQL/train_and_evaluate/data/SynSQL-2.5M/data.json'
    record_count = count_json_records(json_file)
    print(f"记录总数: {record_count}")