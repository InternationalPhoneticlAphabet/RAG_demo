"""
FQA（Frequently Asked Questions）系统 —— 查询主流程入口

一、FQA系统的主要作用
    在RAG系统中充当"快速路径"（Fast Path），作为LLM生成之前的第一道拦截层：
    1. 降低延迟：BM25词频匹配为毫秒级，远快于RAG的向量检索 + LLM推理（数秒级）
    2. 降低成本：命中FQA时不调用LLM，避免按token计费的API开销
    3. 答案可控：高频问题的标准答案由人工维护在MySQL中，不依赖LLM生成的不确定性结果
    4. 减轻RAG压力：减少向量数据库和LLM的调用次数，提升系统整体吞吐量

    本质上是"级联检索"（Cascading Retrieval）思想的实现：先快后慢、先便宜后贵。

二、核心逻辑
    1. 离线阶段：将结构化FQA问答对存入MySQL，对问题进行jieba分词，构建BM25检索器
    2. 在线阶段：
       用户query → 分词 → BM25计算相似度 → softmax归一化 → 双重阈值判断
       ├─ 超过阈值（相对0.85 且 绝对10.0）→ 查Redis缓存 → 未命中则查MySQL → 返回答案并回写Redis
       └─ 未超过阈值 → 交由RAG系统处理（向量检索 + LLM生成）

三、BM25检索的缺点
    1. 语义鸿沟：基于词频统计，无法理解语义。例如"如何重启服务器"和"服务器怎么重启"语义相同，
       但分词结果差异大，BM25匹配分数可能很低
    2. 同义词问题：无法处理同义词，如"Python"和"蟒蛇"、"数据库"和"DB"
    3. 多义词问题：无法区分上下文，如"苹果"可能指水果也可能指公司
    4. 长文本退化：BM25对长文档的区分度下降，文档越长越容易误匹配
    5. 无上下文理解：不考虑词序和句法结构，"猫追狗"和"狗追猫"的BM25分数完全相同

四、常见面试题

    Q1: 如何解决FQA无法区分语义相似度的问题？
    A1: 三种升级方案：
        ① 将BM25替换为embedding语义检索：把FQA问答对向量化存入Milvus，用余弦相似度匹配
        ② 混合检索：BM25关键词匹配 + embedding语义匹配，分数融合（RRFRanker）
        ③ 查询改写：用LLM对用户query进行同义词扩展或改写，再送入BM25检索

    Q2: FQA为什么同时使用MySQL和Redis？
    A2: 分层存储策略，各取所长：
        - MySQL：持久化存储，保存所有FQA问答对数据，保证数据不丢失，支持增删改查
        - Redis：高速缓存，存储最近/最常命中的答案，读取速度为微秒级（内存操作）
        查询优先级：Redis缓存命中 → 直接返回（最快）；未命中 → 查MySQL → 回写Redis
        这样既保证了数据持久性，又通过缓存大幅降低高频问题的查询延迟

    Q3: 如何减少或避免FQA系统的错误命中？
    A3: 错误命中是指BM25匹配到了语义不相关的问题，但分数碰巧超过阈值：
        ① 双重阈值：同时要求归一化分数（相对阈值0.85）和原始分数（绝对阈值10.0）都达标，
           避免单一阈值的盲区（仅用相对阈值可能在所有问题都不相关时仍选中最高分）
        ② 升级为embedding语义匹配：语义向量能更好地区分"真正相似"和"碰巧词频高"
        ③ 增加reranker重排序：BM25粗筛后，用cross-encoder精排，过滤误匹配
        ④ 人工维护问答对质量：确保MySQL中的问题表述覆盖常见的同义表达

    Q4: BM25检索算法的核心思想是什么？
    A4: BM25（Best Matching 25）是TF-IDF的改进版本，核心公式考虑三个因素：
        ① 词频（TF）：词在文档中出现的次数越多，分数越高，但引入饱和函数防止过度奖励
           （一个词出现100次不比10次好10倍）
        ② 逆文档频率（IDF）：在所有文档中越罕见的词，权重越高
           （"的"这种高频词权重低，"Kubernetes"这种专业词权重高）
        ③ 文档长度归一化：短文档中出现一个词，比长文档中出现同样次数的权重更高
        本项目使用rank_bm25库的BM25Okapi实现，对分词后的问题进行相似度打分，
        再通过softmax归一化将分数映射到[0,1]区间，便于阈值判断
"""

# 导入 MySQL 客户端
from db.mysql_client import MysqlClient
# 导入 Redis 客户端
from cache.redis_client import RedisClient
# 导入 BM25 搜索
from retrieval.bm25_search import BM25Search
# 导入日志
from base.logger import logger
# 导入时间库
import time
import sys

class MySQLQASystem:
    # 1.初始化连接
    def __init__(self):
        # 初始化日志
        self.logger = logger
        # 初始化 MySQL 客户端
        self.mysql_client = MysqlClient()
        # 初始化 Redis 客户端
        self.redis_client = RedisClient()
        # 初始化 BM25 搜索
        self.bm25_search = BM25Search(self.redis_client, self.mysql_client)

    # 2.FQA系统查询主流程
    def search(self, query):

        start_time = time.time()
        # 记录查询信息
        self.logger.info(f"处理查询: '{query}'")
        # 1.执行 BM25 搜索, 返回答案
        answer, _ = self.bm25_search.search(query, threshold=(0.85, 10.0))
        if answer:
            # 记录 MySQL 答案
            self.logger.info(f"MySQL 答案: {answer}")
        else:
            # 记录无答案
            self.logger.info("SQL中未找到答案, 需要调用RAG系统")
            # 设置默认答案
            answer = "SQL未找到答案"
        # 计算处理时间
        processing_time = time.time() - start_time
        # 记录处理时间
        self.logger.info(f"查询处理耗时 {processing_time:.2f}秒")
        # 返回答案
        return answer
def main():
    # 初始化 MySQL 系统
    mysql_system = MySQLQASystem()
    try:
        # 打印欢迎信息（放在初始化之后，确保日志输出完毕再显示提示）
        print("\n欢迎使用 MySQL 问答系统！")
        print("输入查询进行问答，输入 'exit' 退出。\n")
        # 刷新stderr，确保初始化日志在输入提示前输出完毕
        sys.stderr.flush()
        while True:
            # 获取用户输入
            query = input("\n输入查询: ").strip()
            if query.lower() == "exit":
                # 记录退出日志
                logger.info("退出 MySQL 系统")
                # 打印退出信息
                print("再见！")
                break
            # 执行查询
            answer = mysql_system.search(query)
            # 打印答案
            print(f"\n答案: {answer}")
    except Exception as e:
        # 记录系统错误
        logger.error(f"系统错误: {e}")
        # 打印错误信息
        print(f"发生错误: {e}")
    finally:
        # 关闭 MySQL 连接
        mysql_system.mysql_client.close()
if __name__ == "__main__":
    # 运行主程序
    main()
