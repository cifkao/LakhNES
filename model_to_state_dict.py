import argparse
import os

import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    args = parser.parse_args()
    
    with open(os.path.join(args.dir, 'model.pt'), 'rb') as f:
        model = torch.load(f)
    with open(os.path.join(args.dir, 'model_state.pt'), 'wb') as f:
        torch.save(model.state_dict(), f)