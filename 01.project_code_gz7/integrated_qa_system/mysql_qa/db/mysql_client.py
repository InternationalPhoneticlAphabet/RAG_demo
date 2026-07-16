"""
mysql_client模块
实现功能:
    1.初始化mysql客户端
    2.创建database
    3.创建高频问答数据表table
    4.写入csv到mysql
    5.读取所有问题
    6.根据问题获取对应的答案

"""
import sys
import os

# 把 integrated_qa_system 目录添加到 Python 搜索路径
# 当前文件在 .../integrated_qa_system/mysql_qa/db/mysql_client.py
# 往上 3 层到达 integrated_qa_system
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pymysql  # pip install pymysql
import pandas as pd
from base.config import config
from base.logger import logger


class MysqlClient:
    # 1.初始化mysql客户端
    def __init__(self):
        try:
            # 1.mysql连接
            self.connect = pymysql.connect(
                host=config.MYSQL_HOST,  # mysql 地址
                port=config.MYSQL_PORT,  # mysql 端口号
                user=config.MYSQL_USER,  # mysql 用户名
                password=config.MYSQL_PASSWORD,  # mysql 密码
            )
            # 2.创建mysql游标, 类比 mysql数据库的执行员，用于执行sql语句并获取结果
            self.cursor = self.connect.cursor()
            # 3.创建database
            self.create_database()
            logger.info("mysql初始化 成功")
        except Exception as e:
            logger.error(f"mysql初始化 失败: {e}")
            raise

    # 2.创建database
    def create_database(self):
        # 1.创建sql语句
        db_name = config.MYSQL_DATABASE
        # 数据库名来自配置文件，无注入风险
        sql = f"CREATE DATABASE IF NOT EXISTS {db_name}"
        try:
            # 2.执行sql,创建 database
            self.cursor.execute(sql)
            # 3.切换到目标数据库
            self.cursor.execute(f"USE {db_name}")
            logger.info(f"创建database {db_name} 成功")
        except Exception as e:
            logger.error(f"创建database {db_name} 失败: {e}")
            raise

    # 3.创建高频问答数据表table
    def create_table(self, table_name='jpkb'):
        # 1.创建sql语句
        # 要保证question字段是唯一的, 因为问题不能重复
        sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} 
        (
            id     INT AUTO_INCREMENT PRIMARY KEY,
            subject_name    VARCHAR(20),
            question        VARCHAR(500) UNIQUE,
            answer          TEXT
        )
        """
        try:
            # 2.执行sql,创建 数据表
            self.cursor.execute(sql)
            # 3.提交，会修改数据
            self.connect.commit()
            logger.info("创建高频问答数据表 成功")
        except Exception as e:
            logger.error(f"创建高频问答数据表 失败: {e}")
            raise

    # 4.写入csv到mysql
    def insert_data(self, csv_path, table_name='jpkb'):
        # 1.读取csv
        df = pd.read_csv(csv_path)
        try:
            # 2.遍历每一行
            for index, row in df.iterrows():
                # 3.获取 学科名称,问题,答案
                question = row["问题"]
                answer = row["答案"]
                subject_name = row["学科名称"]
                # 4.插入数据到mysql
                # 创建sql
                # 方法1：严禁使用！字符串拼接，存在sql注入风险(直接把用户输入的内容拼接到sql语句中)
                # sql = f"""
                # INSERT INTO table_name (question, answer, subject_name) VALUES ('{question}', '{answer}', '{subject_name}')
                # """
                # 比如 输入 question = "a", answer = "a", subject_name = "a; DROP TABLE table_name;"
                # 方法2：参数化查询，使用 %s 占位符，分离 sql语句 和 用户输入，可以防止sql注入攻击
                # 如果 question已经存在，则update answer 和 subject_name
                sql = f"""
                INSERT INTO {table_name} (question, answer, subject_name) VALUES (%s, %s, %s)
                ON DUPLICATE KEY UPDATE
                answer = VALUES(answer),
                subject_name = VALUES(subject_name)
                """

                # 执行sql
                self.cursor.execute(sql, (question, answer, subject_name))

            # 提交,一次性提交
            # 事务：一组操作，确保要么全部成功，要么全部失败
            # 类比：转账，A给B转了100元，A刚转出，A账户-100，B还没有收到，然后断电了；会导致A投诉；
            # 将这个转账操作封装为一个事务，要么全部成功（A转出100元，A账户-100元，B收到100元，B账户+100元），要么全部失败（A和B的账户都不变）
            self.connect.commit()
            logger.info(f"写入高频问答数据表 成功, 写入了 {len(df)} 条数据")
        except Exception as e:
            # 事务回滚，如果事务执行一半出错，则回滚
            self.connect.rollback()
            logger.error(f"写入高频问答数据表 失败: {e}")
            raise

    # 5.读取所有问题
    def fetch_questions(self, table_name='jpkb'):
        try:
            # 1.创建sql
            sql = f"SELECT question FROM {table_name}"

            # 2.执行sql
            self.cursor.execute(sql)

            # 3.获取结果
            results = self.cursor.fetchall()

            # 4.获取questions
            questions = [result[0] for result in results]
            logger.info(f"读取所有问题 成功, 共 {len(questions)} 条数据")
            return questions

        except Exception as e:
            logger.error(f"读取所有问题 失败: {e}")
            raise

    # 6.根据问题获取对应的答案
    def fetch_answer(self, question, table_name='jpkb'):
        try:
            # 1.创建sql
            # 参数化查询，使用 %s 占位符，分离 sql语句 和 用户输入，可以防止sql注入攻击
            sql = f"SELECT answer FROM {table_name} WHERE question = %s"

            # 2.执行sql
            self.cursor.execute(sql, (question,))

            # 3.获取答案
            results = self.cursor.fetchone()

            if results:
                answer = results[0]
                logger.info(f"根据问题获取对应的答案 成功, 问题: {question}")
                return answer
            else:
                logger.info(f"未找到问题对应的答案, 问题: {question}")
                return None

        except Exception as e:
            logger.error(f"根据问题 {question} 获取对应的答案 失败: {e}")
            raise

    # 7.关闭连接
    def close(self):
        try:
            self.cursor.close()
            self.connect.close()
            logger.info("关闭MySQL连接 成功")
        except Exception as e:
            logger.error(f"关闭MySQL连接 失败: {e}")


# 主程序
if __name__ == '__main__':
    # 1.初始化mysql客户端
    mysql_client = MysqlClient()

    # 2.创建高频问答数据表table
    mysql_client.create_table()
    mysql_client.create_table(table_name='plate_faq')

    # 3.写入csv到mysql
    mysql_client.insert_data("../data/JP学科知识问答.csv")
    mysql_client.insert_data('../data/制版厂知识问答.csv', table_name='plate_faq')
    # 4.读取所有问题
    questions = mysql_client.fetch_questions()
    questions2 = mysql_client.fetch_questions(table_name='plate_faq')
    # print(questions)

    # 5.根据问题获取对应的答案
    answer = mysql_client.fetch_answer("pycharm专业版如何激活")
    answer2 = mysql_client.fetch_answer("CTP制版的流程是什么？", table_name='plate_faq')
    print(answer)
    print(answer2)

    # 6.关闭连接
    mysql_client.close()
