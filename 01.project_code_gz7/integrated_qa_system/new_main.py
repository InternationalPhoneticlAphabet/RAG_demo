"""
优化版: 实现FQA和RAG结合的问答系统，用户输入一个query，返回最终的答案。并支持对话历史管理与流式输出。
整体流程：
    1. 初始化系统组件
       1.1 初始化 MySQL 客户端
       1.2 初始化 Redis 客户端
       1.3 初始化 FQA 检索器
       1.4 初始化 RAG 系统
       1.5 初始化 OpenAI 客户端
       1.6 保存日志与配置对象
       1.7 初始化对话表
    2. 调用 DashScope API
       2.1 构造系统提示词
       2.2 发起流式请求
       2.3 逐块收集模型输出
       2.4 异常时返回错误提示
    3. 初始化对话表
       3.1 创建对话历史表
       3.2 提交事务
       3.3 记录初始化成功日志
       3.4 记录初始化失败日志
    4. 获取最近对话历史
       4.1 查询最近 5 轮对话
       4.2 组装历史记录
       4.3 调整为正序，便于上下文拼接
       4.4 记录查询失败日志
    5. 对外获取对话历史
       5.1 直接返回最近历史
    6. 更新对话历史
       6.1 写入新对话
       6.2 重新读取最新历史
       6.3 提交事务
       6.4 记录更新成功日志
       6.5 记录数据库异常并回滚
       6.6 记录未知异常并回滚
    7. 清除对话历史
       7.1 逻辑删除当前对话记录
       7.2 提交事务
       7.3 记录清除成功日志
       7.4 记录清除失败日志
    8. 主流程：问题 -> 答案
       8.1 记录开始时间
       8.2 读取历史对话
       8.3 先走 FQA 检索
       8.4 FQA 命中直接返回
       8.5 需要时再走 RAG
       8.6 调用 RAG 生成答案
       8.7 更新对话历史
       8.8 记录耗时
       8.9 未命中时返回兜底提示
"""
import uuid # 生成对话ID
import pymysql

# 数据库与缓存组件
from mysql_qa.cache.redis_client import RedisClient
from mysql_qa.db.mysql_client import MysqlClient
from mysql_qa.retrieval.bm25_search import BM25Search

# RAG 组件
from rag_qa.core.vector_store import VectorStore
from rag_qa.core.rag_system import RAGSystem
from rag_qa.core.prompts import RAGPrompts

# 配置与日志
from base.config import config
from base.logger import logger

# LLM 客户端
from openai import OpenAI

# 时间统计
import time

# FQA + RAG 集成问答系统
class IntegratedQASystem():
    """整合 FQA 与 RAG 的问答系统。"""

    # 1.初始化系统组件
    def __init__(self):
        # 1.初始化 MySQL 客户端
        self.mysql_client = MysqlClient()
        # 2.初始化 Redis 客户端
        self.redis_client = RedisClient()
        # 3.初始化 FQA 检索器
        self.fqa_bm25search = BM25Search(
            redis_client=self.redis_client,
            mysql_client=self.mysql_client
        )
        # 4.初始化 RAG 系统
        self.rag = RAGSystem(
            vector_store=VectorStore(),
            llm=self.call_dashscope_api,
        )
        # 5.初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=config.DASHSCOPE_API_KEY,
            base_url=config.DASHSCOPE_BASE_URL
        )
        # 6.保存日志与配置对象
        self.logger = logger
        self.config = config
        # 7.初始化对话表
        self.init_conversations_table()

    # 2.调用 DashScope API
    def call_dashscope_api(self, prompt, system_prompt=None):
        try:
            # 1.使用传入的 system_prompt，未传入时默认使用 RAG system prompt
            if system_prompt is None:
                system_prompt = RAGPrompts.rag_system_prompt()

            # 2.发起流式请求
            completion = self.client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                timeout=30,
                stream=True
            )
            # 3.逐块收集模型输出
            # 初始化收集流式输出的字符串
            collected_content = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    # 累积内容
                    collected_content += chunk.choices[0].delta.content
                    yield chunk.choices[0].delta.content
            return collected_content

        except Exception as e:
            # 4.异常时返回错误提示
            logger.error(f"调用 DashScope API 异常: {e}")
            yield f"出问题了,请联系客服: {config.CUSTOMER_SERVICE_PHONE}!"

    # 3.初始化对话表
    def init_conversations_table(self):
        try:
            # 1.创建对话历史表
            self.mysql_client.cursor.execute("""
                                             CREATE TABLE IF NOT EXISTS conversations
                                             (
                                                 id          BIGINT AUTO_INCREMENT PRIMARY KEY,
                                                 session_id  VARCHAR(36)           NOT NULL,
                                                 question    TEXT                  NOT NULL,
                                                 answer      TEXT                  NOT NULL,
                                                 timestamp   DATETIME              NOT NULL,
                                                 _is_deleted BOOLEAN DEFAULT FALSE NOT NULL,
                                                 INDEX idx_session_id (session_id)
                                             )
                                             """)
            # 2.提交事务
            self.mysql_client.connect.commit()
            # 3.记录初始化成功日志
            self.logger.info("对话历史表初始化成功")
        except pymysql.MySQLError as e:
            # 4.记录初始化失败日志
            self.logger.error(f"初始化对话历史表失败: {e}")
            raise

    # 4.获取最近对话历史
    def _fetch_recent_history(self, session_id: str) -> list:
        try:
            # 1.查询最近 5 轮对话，按照时间倒序（从大到小），时间越新/大越靠前
            self.mysql_client.cursor.execute("""
                                             SELECT question, answer
                                             FROM conversations
                                             WHERE session_id = %s
                                               and _is_deleted = FALSE
                                             ORDER BY timestamp DESC
                                             LIMIT %s
                                             """, (session_id, 5))
            # 2.组装历史记录
            history = [{"question": row[0], "answer": row[1]} for row in self.mysql_client.cursor.fetchall()]
            # 3.调整为正序，便于上下文拼接，时间正序（从小到大）
            return history[::-1]
        except pymysql.MySQLError as e:
            # 4.记录查询失败日志
            self.logger.error(f"获取对话历史失败: {e}")
            return []

    # 5.对外获取对话历史
    def get_session_history(self, session_id: str) -> list:
        # 1.直接返回最近历史
        return self._fetch_recent_history(session_id)

    # 6.更新对话历史
    def update_session_history(self, session_id: str, question: str, answer: str) -> list:
        try:
            # 1.写入新对话
            self.mysql_client.cursor.execute("""
                                             INSERT INTO conversations (session_id, question, answer, timestamp)
                                             VALUES (%s, %s, %s, NOW())
                                             """, (session_id, question, answer))
            # 2.重新读取最新历史
            history = self._fetch_recent_history(session_id)
            # 删除超出 5 轮的旧记录
            # TODO：复杂嵌套SQL，从最内层的括号开始看，然后逐渐向外
            # 1. 【取最近5条记录】获取最近5轮对话的ID
            # 2. 【取全集和最近5条记录差集】查询当前会话(session_id)下，id不在 获取最近5轮对话的ID以内的
            # 3. 【删除差集】删除第二步的结果
            # self.mysql_client.cursor.execute("""
            #                                  DELETE
            #                                  FROM conversations
            #                                  WHERE session_id = %s
            #                                    AND id NOT IN (SELECT id
            #                                                   FROM (SELECT id
            #                                                         FROM conversations
            #                                                         WHERE session_id = %s
            #                                                         ORDER BY timestamp DESC
            #                                                         LIMIT %s) AS sub)
            #                                  """, (session_id, session_id, 5))
            # 3.提交事务
            self.mysql_client.connect.commit()
            # 4.记录更新成功日志
            self.logger.info(f"对话 {session_id} 历史更新成功")
            return history
        except pymysql.MySQLError as e:
            # 5.记录数据库异常并回滚
            self.logger.error(f"更新对话历史失败: {e}")
            self.mysql_client.connect.rollback()
            raise
        except Exception as e:
            # 6.记录未知异常并回滚
            self.logger.error(f"更新对话历史意外错误: {e}")
            self.mysql_client.connect.rollback()
            raise

    # 7.清除对话历史
    def clear_session_history(self, session_id: str) -> bool:
        try:
            # 1.逻辑删除当前对话记录
            new_sql = """
                      update conversations
                      set _is_deleted = True
                      WHERE session_id = %s \
                      """
            self.mysql_client.cursor.execute(new_sql, (session_id,))
            # 2.提交事务
            self.mysql_client.connect.commit()
            # 3.记录清除成功日志
            self.logger.info(f"对话 {session_id} 历史已清除")
            return True
        except pymysql.MySQLError as e:
            # 4.记录清除失败日志
            self.logger.error(f"清除对话历史失败: {e}")
            self.mysql_client.connect.rollback()
            return False

    # 8.主流程：问题 -> 答案
    def query(self, query, session_id=None, source_filter=None):
        # 0.添加query校验逻辑，防止为空
        if not query:
            yield "请输入问题!", True
            return
        # 1.记录开始时间
        start_time = time.time()
        # 2.读取历史对话
        history = self._fetch_recent_history(session_id=session_id) if session_id else []
        # 3.先走 FQA 检索
        answer, is_need_rag = self.fqa_bm25search.search(query, threshold=[0.85,10.0])
        # 4.FQA 命中直接返回
        if answer:
            duration = time.time() - start_time
            logger.info("在 FQA 模块获取到了答案，执行时间: {}".format(duration))
            if session_id:
                self.update_session_history(session_id=session_id, question=query, answer=answer)
            yield answer, True
            return

        logger.info(f"FQA 未命中，问题：{query}")

        # 5.需要时再走 RAG
        if is_need_rag:
            # 5.1 调用 RAG 生成答案
            logger.info(f"尝试查询 RAG 模块，问题：{query}")
            collected_answer = ''
            for token in self.rag.generate_answer(query, source_filter=source_filter, history=history):
                collected_answer += token
                yield token, False
            # 5.2 更新对话历史
            if session_id:
                self.update_session_history(session_id=session_id, question=query, answer=collected_answer)
            # 5.3 记录耗时
            duration = time.time() - start_time
            logger.info("在 RAG 系统中获取到了答案，执行时间: {}".format(duration))
            yield '', True
        else:
            # 6.未命中时返回兜底提示
            duration = time.time() - start_time
            logger.info("未能查询到对应的答案: {}".format(duration))
            yield f"未找到对应的答案。请联系客服：{config.CUSTOMER_SERVICE_PHONE}", True

def main():
    # 定义主函数，提供命令行交互界面
    qa_system = IntegratedQASystem()  # 初始化问答系统
    # 生成唯一对话 ID
    session_id = str(uuid.uuid4())
    # 打印欢迎信息
    print("\n欢迎使用集成问答系统！")
    # 打印对话 ID
    print(f"对话ID: {session_id}")
    # 打印支持的学科类别
    print(f"支持的学科类别：{qa_system.config.VALID_SOURCES}")
    # 提示用户输入查询或退出
    print("输入查询进行问答，输入 'exit' 退出。")
    try:
        while True:
            time.sleep(1)
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
            source_filter = input(f"请输入学科类别 ({'/'.join(qa_system.config.VALID_SOURCES)}) (直接回车默认不过滤): ").strip()
            if source_filter and source_filter not in qa_system.config.VALID_SOURCES:
                # 如果学科过滤无效，记录警告日志
                logger.warning(f"无效的学科类别 '{source_filter}'，将不过滤")
                # 设置为空，忽略过滤
                source_filter = None
            # 打印答案提示
            print("\n答案: ", end="", flush=True)
            # 初始化累积答案的字符串
            answer = ""
            # 迭代 query 方法的生成器
            for token, is_complete in qa_system.query(query, source_filter=source_filter, session_id=session_id):
                if token:
                    # 仅当 token 非空时打印
                    print(token, end="", flush=True)
                    # 累积答案
                    answer += token
                if is_complete:
                    # 如果是完整答案或流结束，换行并退出循环
                    print()
                    break
            # 打印对话历史
            history = qa_system.get_session_history(session_id)
            print("\n最近对话历史:")
            for idx, entry in enumerate(history, 1):
                # 按顺序打印历史记录
                print(f"{idx}. 问: {entry['question']}\n   答: {entry['answer']}")
    except Exception as e:
        # 记录系统错误日志
        logger.error(f"系统错误: {e}")
        # 打印错误信息
        print(f"发生错误: {e}")
    finally:
        # 关闭 MySQL 连接
        qa_system.mysql_client.close()

if __name__ == "__main__":
    # 如果脚本作为主程序运行，调用 main 函数
    main()
