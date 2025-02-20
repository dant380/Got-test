import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.addHandler(handler)

logger.debug("Тестовий запуск скрипта")

def main():
    logger.debug("Всередині main()")
    print("Скрипт працює")

if __name__ == "__main__":
    main()