import experiment_from_config
from utils.config.config import CONFIG_PATH_H2

def main():
    experiment_from_config.run(CONFIG_PATH_H2,use_gpu=False)

if __name__ == "__main__":
    main()
