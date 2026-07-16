"""
简化版: 实现FQA和RAG结合的问答系统，用户输入一个query，返回最终的答案
工作流：
    1.初始化方法
        1.初始化 MySQL 客户端
        2.初始化 Redis 客户端
        3.初始化 FQA系统的BM25检索器
        4.初始化 RAG系统
        5.初始化 OpenAI客户端
    2.调用 DashScope API
    3.主流程：问题 -> 答案
        0. 添加query校验逻辑，防止为空
        1. 记录开始时间
        2. 调用FQA系统的BM25Search.query
        3. 如果得到答案，直接返回
        4. 如果需要查询RAG系统，调用RAGSystem.generate_answer
"""
# 导入 MySQL 系统组件，用于数据库操作和搜索
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.db.mysql_client import MysqlClient
from mysql_qa.retrieval.bm25_search import BM25Search

# 导入 RAG 系统组件，用于知识库检索和答案生成
from rag_qa.core.vector_store import VectorStore
from rag_qa.core.rag_system import RAGSystem

# 导入配置和日志工具，用于系统配置和日志记录
from base.config import config
from base.logger import logger
from rag_qa.core.prompts import RAGPrompts

# 导入 OpenAI 客户端，用于调用 DashScope API
from openai import OpenAI

# 导入时间库，用于记录处理时间
import time

# 定义类，实现FQA和RAG结合的问答系统
class IntegratedQASystem():
    """
    这是一个FQA+RAG的问答系统，实现从用户问题query到最终答案answer的全流程
    """
    # 1.初始化方法
    def __init__(self):
        # 1.初始化 MySQL 客户端
        self.mysql_client = MysqlClient()
        # 2.初始化 Redis 客户端
        self.redis_client = RedisClient()
        # 3.初始化 FQA系统的BM25检索器
        self.fqa_bm25search = BM25Search(
            redis_client=self.redis_client,
            mysql_client=self.mysql_client
        )

        # 4.初始化 RAG系统
        self.rag = RAGSystem(
            vector_store=VectorStore(),
            llm=self.call_dashscope_api,
        )

        # 5.初始化 OpenAI客户端
        self.client = OpenAI(
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.DASHSCOPE_BASE_URL
        )

    # 2.调用 DashScope API
    def call_dashscope_api(self, prompt, system_prompt=None):
        """DashScope API 调用，支持传入 system_prompt"""
        try:
            # 1.使用传入的 system_prompt，未传入时默认使用 RAG 系统提示词
            if system_prompt is None:
                system_prompt = RAGPrompts.rag_system_prompt()
            # 2.创建聊天对话
            completion = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            # 3.返回LLM生成的答案
            return completion.choices[
                0].message.content if completion.choices else f"未找到对应答案，请联系客服: {config.CUSTOMER_SERVICE_PHONE}"
        except Exception as e:
            logger.error(f"调用DashScope API异常: {e}")
            return f"出问题了,请联系客服: {config.CUSTOMER_SERVICE_PHONE}!"


    # 3.主流程：问题 -> 答案
    def query(self, query, source_filter=None):
        # 0.添加query校验逻辑，防止为空
        if not query:
            return "请输入问题!"
        # 1. 记录开始时间
        start_time = time.time()
        # 2. 调用FQA系统的BM25Search.search
        answer, is_need_rag = self.fqa_bm25search.search(query, threshold=[0.85, 11.0])

        # 3. 如果得到答案，直接返回
        duration = time.time() - start_time
        if answer:
            logger.info(f"FQA系统获得了答案，耗时: {duration:.2f}s")
            return answer
        logger.info(f"FQA系统未获得答案，耗时: {duration:.2f}s")

        # 4. 如果需要查询RAG系统，调用RAGSystem.generate_answer
        if is_need_rag:
            logger.info(f"开始查询RAG系统, 问题: {query}")
            # generate_answer 是生成器函数(yield from)，需要消耗生成器拿到完整答案
            answer = ''.join(self.rag.generate_answer(query, source_filter=source_filter))
            duration = time.time() - start_time
            logger.info(f"RAG系统获得了答案，耗时: {duration:.2f}s")
            return answer
        else:
            duration = time.time() - start_time
            logger.info(f"RAG系统未获得答案，耗时: {duration:.2f}s")
            return f"未找到对应答案，请联系客服: {config.CUSTOMER_SERVICE_PHONE}"

def main():
    # 定义主函数，提供命令行交互界面
    qa_system = IntegratedQASystem()  # 初始化问答系统
    try:
        # 打印欢迎信息
        print("\n欢迎使用集成问答系统！")
        # 打印支持的学科类别
        print(f"支持的来源: {config.VALID_SOURCES}")
        # 提示用户输入查询或退出
        print("输入查询进行问答，输入 'exit' 退出。")
        while True:
            # 获取用户输入的查询
            query = input("\n输入查询: ").strip()
            if query.lower() == "exit":
                # 如果用户输入 exit，记录退出日志
                logger.info("退出系统")
                # 打印退出信息
                print("再见！")
                # 退出循环
                break
            # 获取用户输入的学科过滤
            source_filter = input(f"输入来源过滤 ({'/'.join(config.VALID_SOURCES)}) (按 Enter 跳过): ").strip()
            if source_filter and source_filter not in config.VALID_SOURCES:
                # 如果学科过滤无效，记录警告日志
                logger.warning(f"无效来源 '{source_filter}'，忽略过滤")
                # 打印无效信息，忽略过滤
                print(f"无效来源 '{source_filter}'，继续无过滤。")
                source_filter = None
            # 执行查询，获取答案
            answer = qa_system.query(query, source_filter)
            # 打印答案
            print(f"\n答案: {answer}")
    except Exception as e:
        # 记录系统错误日志
        logger.error(f"系统错误: {e}")
        # 打印错误信息
        print(f"发生错误: {e}")
    finally:
        # 无论是否发生错误，关闭 MySQL 连接
        qa_system.mysql_client.close()

if __name__ == "__main__":
    # 如果脚本作为主程序运行，调用 main 函数
    main()
