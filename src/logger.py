import logging

# 로거 설정
def setup_logger(name, log_file, level=logging.DEBUG):
    """
    로거 설정
        name: 로거 이름
        log_file: 로그 파일명
        level: 로그 레벨 (기본값 DEBUG)
    """
    handler = logging.FileHandler(log_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# main logger
logger = setup_logger('main_logger', 'main_debug.log')
