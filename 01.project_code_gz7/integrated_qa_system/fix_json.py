import json

file_path = r"D:\HEIMA\workspace\python\pythonProject2\RAG_QA\01.project_code_gz7\integrated_qa_system\data\model_generic.json"

with open(file_path, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

data = [json.loads(line) for line in lines]

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"修复完成！共 {len(data)} 条数据")
