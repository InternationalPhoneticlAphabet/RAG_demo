"""
定义 RAGSystem 类，封装 RAG 系统的核心逻辑
1.初始化方法，设置 RAG 系统的基本参数
2.假设问题检索策略（HyDE）
    1. 获取 问题检索策略 对应提示词模板
    2. 调用大模型生成假设答案
    3. 基于假设答案查询父块作为上下文
3.子查询检索策略
    1.获取子查询检索策略的 Prompt 模板
    2.调用大模型生成子查询列表
    3.使用子查询进行混合检索
    4.对所有检索结果进行去重
4.回溯问题检索策略
    1.获取回溯问题检索策略的 Prompt 模板
    2.调用大模型生成回溯问题
    3.使用回溯问题进行检索
5.动态选择检索策略并整合结果
    1.未指定策略时通过策略选择器选择策略
    2.根据检索策略进行文档检索
    3.截取上下文文档数量（CANDIDATE_M）
6.端到端处理用户查询并生成答案
    1. 使用意图识别模型判断问题类型（通用/专业）
    2. 通用知识，直接调用 LLM 生成答案
    3. 专业咨询：
      3.1 选择最佳检索策略
      3.2 检索合并相关文档
      3.3 构建上下文
      3.4 组合提示模板调用 LLM

"""
import os.path

# core/rag_system.py 源码
# RAGPrompts包含： 1. augment提示词，用于结合query和上下文生成答案；2. 假设问题检索策略、子查询检索策略、回溯问题检索策略 对应的提示词模板
from rag_qa.core.prompts import RAGPrompts
# 导入 time 模块，用于计算时间
import time
from base.config import config
from base.logger import logger

# 区分 专业咨询 和 通用知识
from rag_qa.core.query_classifier import QueryClassifier  # 导入查询分类器
# 将专业咨询进一步分类，做策略选择
from rag_qa.core.strategy_selector import StrategySelector  # 导入策略选择器

# 定义RAGSystem类，实现RAG系统核心逻辑
class RAGSystem:
    # 1.初始化方法，设置 RAG 系统的基本参数
    def __init__(self, vector_store, llm):
        # 1.设置向量数据库对象
        self.vector_store = vector_store

        # 2.设置大模型调用函数
        self.llm = llm

        # 3.获取RAG提示词模板
        self.rag_prompt = RAGPrompts.rag_prompt()

        # 4.初始化查询分类器
        self.query_classifier = QueryClassifier(model_path=os.path.join(config.MODELS_DIR,"bert_query_classifier"))

        # 5.初始化策略选择器
        self.strategy_selector = StrategySelector()

    # 2.假设问题检索策略（HyDE）
    # 获取 假设问题检索策略 的文档检索得到的 上下文
    def _retrieve_with_hyde(self, query):
        logger.info(f"使用 假设问题检索策略, query: {query}")
        # 1. 获取 假设问题检索策略 对应提示词模板
        prompt_template = RAGPrompts.hyde_prompt()
        try:
            # 2. 调用大模型生成假设答案
            # 注意：self.llm 是生成器函数(yield)，需要用 ''.join() 消耗生成器拿到完整字符串
            hypo_answer = ''.join(self.llm(prompt_template.format(query=query))).strip()
            # 3. 基于假设答案 查询父块作为上下文
            # 进行文档检索：混合检索 + 重排
            return self.vector_store.hybrid_search_with_rerank(
                query=hypo_answer, # 这里输入的是假设答案hypo_answer,而不是原始的query，因为假设问题检索策略基于假设答案进行检索
                top_k=config.RETRIEVAL_K,
            )
        except Exception as e:
            logger.error(f"假设问题检索策略 执行错误: {e}")
            return []

    # 3.子查询检索策略
    def _retrieve_with_subqueries(self, query):
        logger.info(f"使用 子查询检索策略, query: {query}")
        # 1.获取子查询检索策略的 Prompt 模板
        prompt_template = RAGPrompts.subquery_prompt()

        try:
            # 2.调用大模型生成子查询列表
            # 注意：self.llm 是生成器函数(yield)，需要用 ''.join() 消耗生成器拿到完整字符串
            subqueries_text = ''.join(self.llm(prompt_template.format(query=query))).strip()
            subqueries = [q.strip() for q in subqueries_text.split("\n") if q.strip()]

            # 3.使用子查询进行混合检索
            # 初始化空列表，存储检索结果
            all_docs = []
            # 遍历每个子查询
            for subquery in subqueries:
                # 1.对每个子查询执行hybrid_search_with_rerank（混合检索 + 重排）
                docs = self.vector_store.hybrid_search_with_rerank(
                    query=subquery,
                    top_k=config.RETRIEVAL_K,
                )

                # 2.添加结果到列表中
                all_docs.extend(docs)
                logger.info(f"子查询: {subquery}, 检索到文档数量: {len(docs)}")

            # 4.对所有检索结果进行去重
            # 基于文档内容 或 ID 进行去重
            unique_docs_dict = {doc.page_content: doc for doc in all_docs}
            unique_docs = list(unique_docs_dict.values())

            logger.info(f"子查询检索策略, 检索到文档的去重后数量: {len(unique_docs)}")
            # 返回去重后的唯一文档
            return unique_docs
        except Exception as e:
            logger.error(f"子查询检索策略 执行错误: {e}")
            return []

    # 4.回溯问题检索策略
    # 返回 回溯问题检索策略 的文档检索得到的 上下文
    def _retrieve_with_backtracking(self, query):
        logger.info(f"使用 回溯问题检索策略, query: {query}")
        # 1.获取回溯问题检索策略的 Prompt 模板
        prompt_template = RAGPrompts.backtracking_prompt()
        try:
            # 2.调用大模型生成回溯问题
            # 注意：self.llm 是生成器函数(yield)，需要用 ''.join() 消耗生成器拿到完整字符串
            backtracking_question = ''.join(self.llm(prompt_template.format(query=query))).strip()
            logger.info(f"生成的回溯问题: {backtracking_question}")

            # 3.使用回溯问题进行检索
            return self.vector_store.hybrid_search_with_rerank(
                query=backtracking_question, # 这里输入的是回溯问题backtracking_question,而不是原始的query，因为回溯问题检索策略基于回溯问题进行检索
                top_k=config.RETRIEVAL_K,
            )
        except Exception as e:
            logger.error(f"回溯问题检索策略 执行错误: {e}")
            return []

    # 5.动态选择检索策略并整合结果
    # 返回 整合后的文档检索结果
    def retrieve_and_merge(self, query, source_filter=None, strategy=None):
        """
        动态选择检索策略并整合结果：
        未指定strategy时，根据query选择检索策略，然后执行对应策略的文档检索，返回检索结果
        :param query: 查询
        :param source_filter: 学科过滤
        :param strategy: 检索策略
        :return: 文档检索结果
        """

        # 1.未指定策略时通过策略选择器选择策略
        if not strategy:
            strategy = self.strategy_selector.select_strategy(query)

        # 2.根据检索策略进行文档检索
        # 初始化检索到的文档列表
        ranked_chunks = []
        if strategy == "假设问题检索":
            ranked_chunks = self._retrieve_with_hyde(query)
        elif strategy == "子查询检索":
            ranked_chunks = self._retrieve_with_subqueries(query)
        elif strategy == "回溯问题检索":
            ranked_chunks = self._retrieve_with_backtracking(query)
        else: # 默认 直接检索
            logger.info(f"使用 直接检索策略, query: {query}")
            ranked_chunks = self.vector_store.hybrid_search_with_rerank(
                query=query,
                top_k=config.RETRIEVAL_K,
                source_filter=source_filter
            )
        logger.info(f"检索策略: {strategy}, 检索到文档数量: {len(ranked_chunks)}")

        # 3.截取上下文文档数量（CANDIDATE_M）
        # 子查询检索策略 的文档数量可以增大
        num_docs = config.CANDIDATE_M if strategy != "子查询检索" else config.CANDIDATE_M*2
        final_context_docs = ranked_chunks[:num_docs]
        logger.info(f"最终的文档数量: {len(final_context_docs)}")
        return final_context_docs

    # 6.端到端处理用户查询并生成答案
    def generate_answer(self, query, history=None, source_filter=None):
        """
        根据用于查询query，调用RAG系统，生成最终答案answer
        :param query: 用户查询
        :param history: 历史对话
        :param source_filter: 学科过滤
        :return: 最终答案
        """
        # 记录开始时间
        start_time = time.time()
        logger.info(f"用户查询: {query}, 学科过滤: {source_filter}")

        # 1. 使用意图识别模型判断问题类型（通用知识 / 专业咨询）
        query_category = self.query_classifier.predict_category(query)
        logger.info(f"问题类型: {query_category}")

        # 2. 通用知识，直接调用 LLM 生成答案
        if query_category == "通用知识":
            logger.info("query为通用知识，直接调用 LLM 生成答案")
            # 直接拼接提示词，不进行文档检索
            if history and isinstance(history, list):
                history_str = "\n\n".join(f"human:{row['question']}; ai:{row['answer']}" for row in history)
            else:
                history_str = ""

            prompt_input = self.rag_prompt.format(
                context="", history=history_str, question=query, phone=config.CUSTOMER_SERVICE_PHONE
            )
            try:
                # self.llm 是生成器函数，用 yield from 逐 token 转发给调用方
                yield from self.llm(prompt_input)
            except Exception as e:
                logger.error(f"直接调用 LLM 执行错误: {e}")
                yield f"抱歉，我无法回答您的问题。请联系人工客服: {config.CUSTOMER_SERVICE_PHONE}"
            process_time = time.time() - start_time
            logger.info(f"通用知识查询完成, 耗时: {process_time}s, 查询：{query}")
            return  # 生成器函数中，return 表示停止生成，不再继续执行后面的代码

        # 3. 专业咨询：
        logger.info("query为专业咨询，执行 RAG 流程")
        # 3.1 选择最佳检索策略
        strategy = self.strategy_selector.select_strategy(query)

        # 3.2 检索合并相关文档
        # list[Doc]
        context_docs = self.retrieve_and_merge(
            query, source_filter=source_filter, strategy=strategy)

        # 3.3 构建上下文
        if context_docs:
            # 兼容 Document / dict / str 三种结构，避免类型不一致导致崩溃
            context_parts = []
            for doc in context_docs:
                if hasattr(doc, "page_content"):
                    context_parts.append(doc.page_content)
                elif isinstance(doc, dict):
                    context_parts.append(doc.get("page_content") or doc.get("text") or doc.get("parent_content") or "")
                else:
                    context_parts.append(str(doc))
            context = "\n\n".join([p for p in context_parts if p])
            logger.info(f"构建上下文完成, 文档数量: {len(context_docs)}")
        else:
            context = ""
            logger.info("没有检索到相关文档, 上下文为空")

        # 3.4 组合提示模板调用 LLM
        # 准备历史对话
        # 验证历史格式：[{}]
        if history and not isinstance(history, list):
            logger.warning(f"无效的历史格式: {type(history)}，忽略历史")
            history = []
            history_str = ""
        elif history:
            history_str = "\n\n".join(f"human:{row['question']}; ai:{row['answer']}" for row in history)
        else:
            history_str = ""
        # 构造 prompt
        prompt_input = self.rag_prompt.format(
            context=context,
            history=history_str,
            question=query,
            phone=config.CUSTOMER_SERVICE_PHONE
        )
        # logger.info(f"最终组合的提示词: {prompt_input}")
        # 调用 LLM
        try:
            # self.llm 是生成器函数，用 yield from 逐 token 转发给调用方
            yield from self.llm(prompt_input)
        except Exception as e:
            logger.error(f"RAG流程调用 LLM 执行错误: {e}")
            yield f"抱歉，我无法回答您的问题。请联系人工客服: {config.CUSTOMER_SERVICE_PHONE}"
        # 记录查询日志
        process_time = time.time() - start_time
        logger.info(f"专业咨询查询完成, 耗时: {process_time}s, 查询：{query}")


if __name__ == "__main__":
    from rag_qa.core.vector_store import VectorStore
    from rag_qa.core.rag_system import RAGSystem

    # 1. 初始化向量库
    vector_store = VectorStore()


    # 2. 定义 LLM 调用函数（模拟）
    def mock_llm(prompt):
        yield "这是一个模拟的 AI 回答。RAG 系统已正常工作！"


    # 3. 创建 RAG 系统
    rag = RAGSystem(vector_store=vector_store, llm=mock_llm)

    # 4. 测试查询
    query = "大模型学什么？"
    print(f"查询: {query}")
    print("-" * 50)

    # 5. 调用生成答案
    for chunk in rag.generate_answer(query):
        print(chunk, end="")