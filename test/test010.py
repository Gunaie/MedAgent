from utils import get_logger

# 测试日志分级
logger = get_logger("medagent.test")
logger.debug("This is DEBUG")
logger.info("This is INFO")
logger.warning("This is WARNING")
logger.error("This is ERROR")

# 检查日志文件是否生成
import os
print(f"Log file exists: {os.path.exists('./logs/medagent.log')}")