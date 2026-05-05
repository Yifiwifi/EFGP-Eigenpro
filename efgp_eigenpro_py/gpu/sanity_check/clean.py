import json

# 替换成你那个卡住的文件名
input_file = "v5_mat32_for_precompute.ipynb" 
output_file = "cleaned_version.ipynb"

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 遍历所有单元格，清空输出
for cell in data.get('cells', []):
    if cell.get('cell_type') == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=1)

print("清理完成！请尝试打开 cleaned_version.ipynb")