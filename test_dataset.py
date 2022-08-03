from datasets.ucf101_decord import UCF101_decord
import torch

if __name__ == '__main__':

    from utils.parser import parse_args, load_config
    from tqdm import tqdm
    config = load_config(args)
    config.DATA.PATH_TO_DATA_DIR = "/home/shared/UCF/splits_classification"
    config.DATA.PATH_PREFIX = "/home/shared/UCF/videos"
    dataset = UCF101_decord(cfg=config, mode="train", num_retries=10)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=4)
    print(f"Loaded train dataset of length: {len(dataset)}")
    for idx, i in enumerate(dataloader):
        print(idx, i[0].shape, i[1:])
        if idx > 2:
            break

    test_dataset = UCF101_decord(cfg=config, mode="val", num_retries=10)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=4)
    print(f"Loaded test dataset of length: {len(test_dataset)}")
    for idx, i in enumerate(test_dataloader):
        print(idx, i[0].shape, i[1:])
        if idx > 2:
            break