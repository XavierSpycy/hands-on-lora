import argparse
import warnings
import yaml

from qwen2ner import Qwen2NERPipeline

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, default="train_config.yaml")

def main():
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    pipeline = Qwen2NERPipeline(config['model_name_or_path'])
    pipeline.run(config)

if __name__ == "__main__":
    main()