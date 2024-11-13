import logging

# 공통 핸들러와 포맷터 설정
file_handler = logging.FileHandler('main_debug.log', mode='w')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# recipe_logger 설정
logger_recipe = logging.getLogger('recipe_logger')
logger_recipe.setLevel(logging.DEBUG)
logger_recipe.addHandler(file_handler)

# main_logger 설정
logger_main = logging.getLogger('main_logger')
logger_main.setLevel(logging.DEBUG)
logger_main.addHandler(file_handler)
