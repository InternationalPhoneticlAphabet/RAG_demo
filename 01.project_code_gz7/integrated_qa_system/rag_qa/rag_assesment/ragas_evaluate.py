"""
用 RAGAS 对 RAG 系统做自动化评估。

流程说明（6步）：
1. 读取评估数据（JSON 文件）
2. 转成 RAGAS 要求的数据格式（Dataset）
3. 初始化评估所需模型（LLM + Embedding）
4. 计算评估指标
5. 打印并保存评估结果
"""

import json
import os
import sys
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness, # 忠实度
    AnswerRelevancy, # 答案相关度
    ContextPrecision, # 上下文相关度
    ContextRecall # 上下文召回
)
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings

# 添加项目根目录到 sys.path，确保模块导入正常
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from base.config import config
from rag_qa.rag_assesment.generate_eval_data import generate_rag_eval_data


# 0) 自动生成 RAG 评估数据集（调用真实 RAG 系统）
print("开始生成 RAG 评估数据集...")
data_file = os.path.join(SCRIPT_DIR, "rag_evaluate_data_real.json")
# 如果data_file不存在，则生成
if not os.path.exists(data_file):
    data_file = generate_rag_eval_data()
    print(f"RAG 评估数据集已生成: {data_file}")
else:
    print("RAG 评估数据集已存在")

# 1) 读取评估数据（RAG 系统真实输出的 question、contexts、answer、ground_truth）
with open(data_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 3) 转成 RAGAS 需要的字段结构
# 注意：字段名要与评估指标要求一致
eval_data = {
    "question": [item["question"] for item in data],          # 问题列表
    "answer": [item["answer"] for item in data],              # 模型回答列表
    "contexts": [item["contexts"] for item in data],           # 上下文列表（list[str]）
    "ground_truth": [item["ground_truth"] for item in data]   # 真实答案列表
}
dataset = Dataset.from_dict(eval_data)


# 4) 初始化 LLM（用于部分指标的判分推理）
# qwen-plus
llm = ChatOpenAI(
    model_name=config.LLM_MODEL,
    openai_api_base=config.DASHSCOPE_BASE_URL,
    openai_api_key=config.DASHSCOPE_API_KEY,
    temperature=0  # 固定输出，减少随机性
)

# 5) 初始化 Embedding（用于语义相似度类指标）
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=config.DASHSCOPE_API_KEY
)


# 6) 执行评估
# 指标含义：
# - Faithfulness：回答是否忠于给定上下文
# - AnswerRelevancy：回答与问题是否相关
# - ContextPrecision：上下文中无关内容是否少
# - ContextRecall：上下文是否覆盖真实答案所需信息
result = evaluate(
    dataset=dataset,
    metrics=[
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall()
    ],
    llm=llm,
    embeddings=embeddings
)


# 6) 输出并保存结果
print("RAGAS 评估结果：")
print(result)

# 保存为 CSV，便于后续查看和汇报（使用绝对路径）
result_csv = os.path.join(SCRIPT_DIR, "ragas_evaluation_results.csv")
result_df = pd.DataFrame([result])
result_df.to_csv(result_csv, index=False)
print(f"评估结果已保存到: {result_csv}")

