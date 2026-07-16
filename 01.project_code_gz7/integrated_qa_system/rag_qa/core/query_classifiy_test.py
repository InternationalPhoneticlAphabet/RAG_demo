import json
from query_classifier import QueryClassifier
from base.config import config
import os

model = QueryClassifier(model_path=os.path.join(config.MODELS_DIR, "bert_query_classifier"))

with open('../../data/test_set.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

correct = 0
errors = []

for item in test_data:
    pred = model.predict_category(item['query'])
    if pred == item['label']:
        correct += 1
    else:
        errors.append((item['query'], pred, item['label']))

print(f"准确率: {correct}/{len(test_data)} = {correct/len(test_data)*100:.1f}%")
print(f"\n错误案例 ({len(errors)}条):")
for q, p, e in errors:
    print(f"  ❌ {q} -> 预测:{p} 期望:{e}")
