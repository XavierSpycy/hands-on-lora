import argparse
import warnings

from qwen2ner import Qwen2NERPipeline

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str)

def main():
    args = parser.parse_args()
    pipeline = Qwen2NERPipeline(args.model_name_or_path)
    pipeline()

if __name__ == "__main__":
    main()