import logging
import os
# 导入配置类
from base.config import config

# logger.py是一个公共模块，用于在项目中所有的代码逻辑中，加入日志打印。
def setup_logger(name, log_file='logs/app.log'):
    """
    构建一个通用的logger对象
    :param name:        logger的名字
    :param log_file:    日志存放的位置。 log_file: 1.从配置文件读取(物理机部署) 2.根据项目启动的根目录计算的相对路径(容器启动)
    :return: logger
    """

    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 设置最低级别

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # 定义日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(module)s %(lineno)d- %(message)s')

    # 设置处理器格式
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # 添加处理器（避免重复添加）
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger

# 单例对象
logger = setup_logger("edu_rag", log_file=config.LOG_FILE)
