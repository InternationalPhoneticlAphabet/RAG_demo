"""
定义redis客户端，实现以下功能：
1.初始化连接
2.get_data: 读取redis中key对应的value
3.set_data: 写入redis
4.get_answer: 根据问题读取答案, key="answer:{question}"。
5.set_answer: 写入问题和答案{key: value}, 规定 key="answer:{question}"。
示例：问题和答案{key: value}的数据示例为 {"answer:大模型学什么": "大模型学大模型"}

"""
import sys
import os

# 把 integrated_qa_system 目录添加到 Python 搜索路径
# 当前文件在 .../integrated_qa_system/mysql_qa/db/mysql_client.py
# 往上 3 层到达 integrated_qa_system
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from redis import StrictRedis
from base.logger import logger
from base.config import config
import json

class RedisClient():
    """
    Redis 缓存客户端，支持：
    - 通用JSON的存取
    - 问答对的存取（自动24h过期）
    """
    # 问答对缓存的默认过期时间：24h
    ANSWER_EXPIRE_SECONDS = 24 * 60 * 60
    def __init__(self):
        """
        初始化 Redis连接
        参数使用默认值即可连接本地 Docker的 Redis

        """
        self.redis = StrictRedis(
            host=config.REDIS_HOST, # 默认连接本地
            port=config.REDIS_PORT, # 默认端口
            db=config.REDIS_DB, # 默认数据库
            decode_responses=True, # 返回字符串，无需手动解码
            encoding='utf-8', # 编码方式
            password=config.REDIS_PASSWORD
        )
        logger.info("redis初始化 成功")

    def get_data(self, key):
        """
        读取 key 的 json数据并反序列化为 python对象
        :param key: redis中的 key
        :return: python对象
        """
        try:
            # 1.获取key对应的value
            value = self.redis.get(key)
            if value is None:
                logger.info(f"读取数据未命中: key={key}")
                return None
            # 2.JSON 字符串 -> Python 对象
            result = json.loads(value)
            logger.info(f"读取数据成功: key={key}")
            return result
        except Exception as e:
            logger.error(f"获取数据失败: key={key}, error={e}")
            return None

    def set_data(self, key, value):
        """
        将 value 转为 字符串，写入 Redis
        :param key: # redis中的 key
        :param value: # python对象
        :return: #  None
        """
        try:
            # 1.Python 对象 -> JSON 字符串
            value = json.dumps(value, ensure_ascii=False)
            # 2.写入redis
            self.redis.set(key, value)
            logger.info(f"写入数据成功: key={key}")
        except Exception as e:
            logger.error(f"写入数据失败: key={key}, error={e}")

    def get_answer(self, question):
        """
        根据问题获取缓存答案
        :param question: # 问题
        :return: # 答案
        """
        try:
            # 1.构造key
            key = f"answer:{question}"
            # 2.获取key对应的answer
            answer = self.redis.get(key)
            if answer is None:
                logger.info(f"读取答案未命中:  key={key}")
                return None
            # 3.设置过期时长
            self.redis.expire(key, self.ANSWER_EXPIRE_SECONDS)
            logger.info(f"读取答案成功: key={key}, answer={answer}")
            # 4.返回answer
            return answer
        except Exception as e:
            logger.error(f"获取答案失败: question={question}, error={e}")
            return None

    def set_answer(self, question, answer):
        """
        写入问题答案
        :param question: # 问题
        :param answer: # 答案
        :return: # None
        """
        try:
            # 1.构造key
            key = f"answer:{question}"
            # 2.写入答案
            self.redis.set(key, answer, ex=self.ANSWER_EXPIRE_SECONDS)
            logger.info(f"写入答案成功: key={key}, answer={answer}")
        except Exception as e:
            logger.error(f"写入答案失败: question={question}, answer={answer}, error={e}")

# 主程序
if __name__ == '__main__':
    redis = RedisClient()
    # 获取问题的答案
    answer = redis.get_answer("地球围绕什么转")
    print(answer)