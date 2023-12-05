import logging
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("unet_log.txt")
formatter = logging.Formatter('%(asctime)s - %(filename)s - %(module)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
