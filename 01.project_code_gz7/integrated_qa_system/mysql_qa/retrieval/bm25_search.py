"""
BM25 检索分层工作流：

1. 数据与资源初始化层
    1.初始化 MySQL 连接
    2.初始化 Redis 连接

2. 数据加载与建模层
    1.从mysql中读取所有问题
    2.对所有问题进行预处理
    3.构建 BM25 检索模型

3. 用户查询与缓存命中层
    1. 判断query是否合法，非空字符串
    2. 先查redis中是否有一样的问题
    3. 对query分词
    4. 用bm25计算query和所有问题的相似度
    5. 对相似度分数进行softmax归一化
    6. 取最大相似度分数，包括 原始分数 和 归一化分数
    7. 双重阈值判断。包括 相对阈值0.85 和 绝对阈值10.0
    8. 根据索引找到对应的原始问题（查redis）
    9. 查看redis中是否有该问题的答案
    10. 查看mysql中是否有该问题的答案
    11. 返回答案，并写入redis
"""

# retrieval/bm25_search.py
# 导入 BM25 算法
from rank_bm25 import BM25Okapi
# 导入数值计算库
import numpy as np
# 导入文本预处理
from mysql_qa.utils.preprocess import preprocess_text
# 导入日志
from base.logger import logger

from mysql_qa.db.mysql_client import MysqlClient
from mysql_qa.cache.redis_client import RedisClient

# 初始化问题名称
# ORIGIN_QUESTION_KEY = "edurag:origin_questions"
# QUESTION_KEY = "edurag:questions"

class BM25Search:
    # 1.初始化
    def __init__(self, redis_client: RedisClient, mysql_client: MysqlClient):
        # 1.初始化日志
        self.logger = logger
        # 2.初始化 redis客户端
        self.redis_client = redis_client
        # 3.初始化 mysql客户端
        self.mysql_client = mysql_client
        # 4.初始化 bm25 模型
        self.bm25 = None
        # 5.初始化 原始问题列表(没有分词)
        self.origin_questions = None
        # 6.初始化 分词后的问题列表
        self.questions = None
        # 7.加载数据
        self._load_data()

    # 2.加载数据，mysql -> redis -> bm25
    def _load_data(self):
        """
        实现FQA 的加载数据功能
        1.判断系统是不是第一次启动
        2.如果是第一次启动
            1.从mysql中读取所有问题
            2.所有问题列表origin_questions 写入redis
            3.分词后的问题列表questions 写入redis
        3.用分词后的问题列表 构建bm25检索器
        :return: None
        """
        # 1.判断系统是不是第一次启动
        # 判断 origin_questions, questions 是否都存在
        # origin_questions = self.redis_client.get_data(ORIGIN_QUESTION_KEY)
        # questions = self.redis_client.get_data(QUESTION_KEY)
        # # 2.如果是第一次启动
        # if not origin_questions or not questions:
        self.logger.info("系统第一次启动，正在加载数据...")
        # 1.从mysql中读取所有问题
        origin_questions = self.mysql_client.fetch_questions()
        if not origin_questions:
            self.logger.error("mysql没有问题数据")
            raise Exception("mysql没有问题数据")
        else:
            self.logger.info(f"从mysql中获取了问题，问题数量：{len(origin_questions)}")
        # # 2.所有问题列表origin_questions 写入redis
        # self.redis_client.set_data(ORIGIN_QUESTION_KEY, origin_questions)
        # self.logger.info(f"所有问题写入redis成功，问题数量：{len(origin_questions)}")
        # 3.分词后的问题列表questions 写入redis
        questions = [preprocess_text(question) for question in origin_questions]
        # self.redis_client.set_data(QUESTION_KEY, questions)
        self.logger.info(f"分词后的问题写入redis成功，问题数量：{len(questions)}")

        # 3.用分词后的问题列表 构建bm25检索器
        self.origin_questions = origin_questions
        self.questions = questions
        self.bm25 = BM25Okapi(self.questions)
        self.logger.info("BM25模型构建成功")

    # 3.查询问题
    def query(self, query, threshold=[0.85, 10.0]):
        """
        实现 FQA 查询：用户输入一个query,进行bm25相似度检索，返回相似度超过双重阈值的问题对应的答案
        1. 判断query是否合法，非空字符串
        2. 先查redis中是否有一样的问题
        3. 对query分词
        4. 用bm25计算query和所有问题的相似度
        5. 对相似度分数进行softmax归一化
        6. 取最大相似度分数，包括 原始分数 和 归一化分数
        7. 双重阈值判断。包括 相对阈值0.85 和 绝对阈值10.0
        8. 根据索引找到对应的原始问题（查redis）
        9. 查看redis中是否有该问题的答案
        10. 查看mysql中是否有该问题的答案
        11. 返回答案，并写入redis
        :param query: 用户输入问题
        :param threshold: 阈值，包含相对阈值和绝对阈值
        :return: answer, True/False, 最相似的问题对应的答案, 是否调用RAG系统
        """
        # 1. 判断query是否合法，非空字符串
        if not isinstance(query, str) or not query.strip():
            self.logger.info(f"用户输入的query非法: {query}")
            # query非法，不需要进入RAG系统
            return None, False

        # 2. 先查redis中是否有一样的问题
        answer = self.redis_client.get_answer(query)
        if answer:
            self.logger.info(f"在redis中找到了一样问题: {query}, 答案: {answer}")
            # 在redis中找到了一样的问题对应的答案，不需要进入RAG系统
            return answer, False

        # 3. 对query分词
        query_tokens = preprocess_text(query)

        # 4. 用bm25计算query和所有问题的相似度
        # 相似度分数形状 1D:(len(questions),)
        scores = self.bm25.get_scores(query_tokens)

        # 5. 对相似度分数进行softmax归一化
        scores_softmax = self._soft_max(scores)

        # 6. 取最大相似度分数，包括 原始分数 和 归一化分数
        max_index = np.argmax(scores_softmax)
        max_score = scores[max_index]
        max_score_softmax = scores_softmax[max_index]
        self.logger.info(f"最大相似度分数: 原始分数：{max_score}, 归一化分数: {max_score_softmax}")

        # 8. 根据索引找到对应的原始问题（查redis）
        origin_question = self.origin_questions[max_index]
        self.logger.info(f"最相似问题: {origin_question}")

        # 7. 双重阈值判断。包括 相对阈值0.85 和 绝对阈值10.0
        if max_score_softmax > threshold[0] and max_score > threshold[1]:
            # 9. 查看redis中是否有该问题的答案
            answer = self.redis_client.get_answer(origin_question)
            if answer:
                self.logger.info(f"在redis中找到了答案: {answer}")
                # 在redis中找到了相似度最高的问题的答案，不需要进入RAG系统
                return answer, False

            # 10. 查看mysql中是否有该问题的答案
            answer = self.mysql_client.fetch_answer(origin_question)
            if answer:
                self.logger.info(f"在mysql中找到了答案: {answer}")
                # 11.返回答案，并回写答案到redis
                self.redis_client.set_answer(origin_question, answer)
                # 在mysql中找到了相似度最高的问题的答案，不需要进入RAG系统
                return answer, False
            else:
                self.logger.info(f"在mysql中未找到答案")
                # 在mysql中未找到相似度最高问题的答案，需要进入RAG系统
                return None, True

        # 12.如果最大相似度分数小于阈值，则返回None,True，表示问题合法，需要进入RAG系统
        return None, True

    def _soft_max(self, scores):
        # 1.指数运算前减去最大值，避免数值溢出
        exp_scores = np.exp(scores - np.max(scores))
        # 2.归一化
        return exp_scores / np.sum(exp_scores)

    def search(self, query, threshold=(0.85, 10.0)):
        return self.query(query, threshold=list(threshold))

# 主程序
if __name__ == '__main__':
    bm25_search = BM25Search(RedisClient(), MysqlClient())
    answer, is_rag = bm25_search.query("如何在 Ubunt 创建VScode快捷方式？", threshold=[0.85, 18.0])
    print(f"答案: {answer}, 是否调用RAG系统: {is_rag}")
