import nltk

# 你希望优先使用的路径
preferred_path = '/mnt/cache/tonghao/envs/omnisql_process_data/lib/nltk_data'

# 如果已经在列表中，先移除
if preferred_path in nltk.data.path:
    nltk.data.path.remove(preferred_path)

# 插入到最前面
nltk.data.path.insert(0, preferred_path)

print("当前 NLTK 路径优先数据")
# 查看当前优先顺序
print(nltk.data.path)

try:
    print("文件已存在，文件在该目录下：")
    print(nltk.data.find('tokenizers/punkt'))
except LookupError:
    print("开始下载 NLTK 数据")
    # print(nltk.data.path)
    nltk.download('punkt')
    try:
        print("下载成功，文件在该目录下：")
        print(nltk.data.find('tokenizers/punkt'))
    except Exception as e:
        print("未发现下载后的文件信息，下载失败")