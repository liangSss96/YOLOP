from lib.dataset import BddDataset

if __name__ == "__main__":
    from lib.config import cfg
    print(cfg.TRAIN.SINGLE_CLS)
    dataset = BddDataset(cfg, True, [640, 640])
    print(len(dataset))