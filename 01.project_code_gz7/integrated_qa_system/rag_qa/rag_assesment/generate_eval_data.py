"""
调用 RAG 系统生成评估数据集 rag_evaluate_data_real.json

流程说明：
1. 读取原始评估数据 rag_evaluate_data.json（仅使用 question 和 ground_truth）
2. 对每个 question，调用真实 RAG 系统获取：
   - contexts：向量检索命中的文档（来自 Milvus 混合检索 + 重排）
   - answer：LLM 基于上下文生成的答案（来自 DashScope API）
3. 将 question + contexts + answer + ground_truth 写入 rag_evaluate_data_real.json
4. 该文件可供 ragas_evaluate.py 做 RAGAS 评估，反映 RAG 系统的真实表现

注意：
- 不修改原始 rag_evaluate_data.json
- contexts 和 answer 完全来自 RAG 系统的实际输出
- 如果检索为空，contexts 为 []，answer 可能为 LLM 自由生成或兜底回复
"""

import json
import os
import sys
import time

# 添加项目根目录到 sys.path，确保模块导入正常
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from base.config import config
from base.logger import logger
from rag_qa.core.vector_store import VectorStore
from rag_qa.core.rag_system import RAGSystem
from rag_qa.core.prompts import RAGPrompts
from openai import OpenAI


# 1. 初始化 RAG 系统
def init_rag_system():
    """初始化向量库和 RAG 系统"""
    logger.info("=" * 50)
    logger.info("正在初始化 RAG 系统...")
    logger.info("=" * 50)

    client = OpenAI(
        api_key=config.DASHSCOPE_API_KEY,
        base_url=config.DASHSCOPE_BASE_URL
    )

    def call_dashscope_api(prompt, system_prompt=None):
        """DashScope API 调用，支持传入 system_prompt"""
        try:
            if system_prompt is None:
                system_prompt = RAGPrompts.rag_system_prompt()
            completion = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            return completion.choices[0].message.content if completion.choices else ""
        except Exception as e:
            logger.error(f"调用 DashScope API 异常: {e}")
            return f"出问题了,请联系客服: {config.CUSTOMER_SERVICE_PHONE}!"

    vector_store = VectorStore()
    rag_system = RAGSystem(
        vector_store=vector_store,
        llm=call_dashscope_api,
    )
    logger.info("RAG 系统初始化完成")
    return rag_system


# 2. 对单个问题执行 RAG 流程
def process_question(rag_system, question):
    """
    对单个问题执行完整的 RAG 流程，返回真实的 contexts 和 answer。
    评估时始终执行检索（不依赖 BERT 分类），因为评估数据集的问题
    全部来自知识库覆盖范围，目的是测试 RAG 管道的检索+生成能力。
    :param rag_system: RAGSystem 实例
    :param question: 用户问题
    :return: (contexts, answer)
    """
    # 2.1 始终执行检索（评估目的：测试完整 RAG 管道）
    # 如果按 BERT 分类跳过检索，"通用知识"类问题会得到空 contexts，
    # 导致 Faithfulness/ContextRecall/ContextPrecision 全部失真
    strategy = rag_system.strategy_selector.select_strategy(question)
    logger.info(f"问题: '{question}' → 检索策略: {strategy}")

    context_docs = rag_system.retrieve_and_merge(question, strategy=strategy)
    contexts = []
    if context_docs:
        for doc in context_docs:
            if hasattr(doc, "page_content"):
                contexts.append(doc.page_content)
            elif isinstance(doc, dict):
                contexts.append(doc.get("page_content", ""))
            else:
                contexts.append(str(doc))
        logger.info(f"检索到 {len(contexts)} 个上下文文档")
    else:
        logger.info("未检索到相关文档")

    # 2.2 基于检索到的上下文生成答案
    if contexts:
        context_text = "\n\n".join(contexts)
        prompt_input = rag_system.rag_prompt.format(
            context=context_text,
            history="",
            question=question,
            phone=config.CUSTOMER_SERVICE_PHONE
        )
        # 消耗生成器，收集完整回答
        llm_result = rag_system.llm(prompt_input)
        if isinstance(llm_result, str):
            answer = llm_result
        else:
            answer = ''.join(llm_result)
        logger.info(f"基于上下文生成答案，长度: {len(answer)} 字符")
    else:
        # 检索为空时，让 RAG 系统自然处理：
        # - 通用知识类query → general_prompt + general_system_prompt → LLM 直接回答
        # - 专业咨询类query → rag_prompt(空context) + rag_system_prompt → "信息不足"
        answer = ''.join(rag_system.generate_answer(question))
        logger.info(f"无上下文，由 RAG 系统自然处理，长度: {len(answer)} 字符")

    return contexts, answer


# 3. 生成 RAG 评估数据集
def generate_rag_eval_data():
    """
    生成 RAG 评估数据集：读取原始问题，调用 RAG 系统获取 contexts 和 answer，
    输出 rag_evaluate_data_real.json 供 RAGAS 评估使用。
    :return: 输出文件路径（str）
    """
    # 3.1 读取原始评估数据（仅使用 question 和 ground_truth）
    input_file = os.path.join(SCRIPT_DIR, "rag_evaluate_data.json")
    output_file = os.path.join(SCRIPT_DIR, "rag_evaluate_data_real.json")

    with open(input_file, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    logger.info(f"读取 {len(original_data)} 个评估问题，来自: {input_file}")

    # 3.2 初始化 RAG 系统
    rag_system = init_rag_system()

    # 3.3 逐个问题调用 RAG 系统
    eval_data = []
    success_count = 0
    fail_count = 0

    for i, item in enumerate(original_data):
        question = item["question"]
        ground_truth = item["ground_truth"]

        logger.info(f"\n{'='*50}")
        logger.info(f"[{i+1}/{len(original_data)}] 处理问题: {question}")
        logger.info(f"{'='*50}")

        start_time = time.time()
        try:
            contexts, answer = process_question(rag_system, question)
            duration = time.time() - start_time
            logger.info(f"处理完成, 耗时: {duration:.2f}s")
            success_count += 1
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"处理失败, 耗时: {duration:.2f}s, 错误: {e}")
            contexts = []
            answer = ""
            fail_count += 1

        eval_data.append({
            "question": question,
            "contexts": contexts,
            "answer": answer,
            "ground_truth": ground_truth
        })

        # 每个问题处理完后保存一次（防止中断丢失数据）
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)

    # 3.4 输出统计 + 诊断信息
    has_context = sum(1 for item in eval_data if item["contexts"])
    no_context = sum(1 for item in eval_data if not item["contexts"])

    logger.info(f"\n{'='*50}")
    logger.info(f"评估数据生成完成！")
    logger.info(f"总计: {len(original_data)} 个问题")
    logger.info(f"成功: {success_count}, 失败: {fail_count}")
    logger.info(f"有上下文: {has_context}, 无上下文: {no_context}")
    if no_context > 0:
        logger.warning(f"以下问题未检索到上下文（可能影响评估指标）:")
        for item in eval_data:
            if not item["contexts"]:
                logger.warning(f"  - {item['question']}")
    logger.info(f"数据已保存到: {output_file}")
    logger.info(f"{'='*50}")

    return output_file


def main():
    generate_rag_eval_data()


if __name__ == "__main__":
    main()