import experiment_from_config
from utils.config.config import CONFIG_PATH_H2O

def main():
    experiment_from_config.run(CONFIG_PATH_H2O,use_gpu=True)

if __name__ == "__main__":
    main()
