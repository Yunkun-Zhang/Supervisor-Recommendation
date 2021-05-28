from models import Net
from process_data import load_data
import argparse

ap = argparse.ArgumentParser()
args = ap.parse_args()


if __name__ == '__main__':
    data = load_data()
