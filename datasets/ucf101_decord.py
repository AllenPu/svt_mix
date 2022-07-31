import os
import numpy as np
from numpy.lib.function_base import disp
import torch
import decord
from PIL import Image
from torchvision import transforms
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset

class UCF101(torch.utils.data.Dataset):
    """
    UCF101 video loader. Construct the UCF101 video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the UCF101 video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train mode, the data loader will take data from the
                train set, and sample one clip per video. For the val and
                test mode, the data loader will take data from relevent set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in ["train", "val", "test"], "Split '{}' not supported for UCF101".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        self._split_idx = mode
        # For training mode, one single clip is sampled from every video. For validation or testing, NUM_ENSEMBLE_VIEWS
        # clips are sampled from every video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from the frames.
        if self.mode in ["train"]:
            self._num_clips = 1
        elif self.mode in ["val", "test"]:
            self._num_clips = (
                    cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        print("Constructing UCF101 {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        '''
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "ucf101_{}_split_1_videos.txt".format(self.mode)
        )
        '''
        ###
        # new added for ucf 101 
        ###
        path_to_file_list = []
        #
        #
        for i in range(1,4):       
            path_to_file = os.path.join(
                    self.cfg.DATA.PATH_TO_DATA_DIR, "{}list0{}.txt".format(self.mode, i)
                )
            path_to_file_list.append(path_to_file)

        #path_to_file_list = [path_to_file1, path_to_file2, path_to_file3]
        
        # check if the file exists
        for path_to_file in path_to_file_list:
            assert os.path.exists(path_to_file), "{} dir not found".format(
                path_to_file
            )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []

        for path_to_file in path_to_file_list:
            with open(path_to_file, "r") as f:
                for clip_idx, path_label in enumerate(f.read().splitlines()):
                    assert (
                            len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
                            == 2
                    )
                    path, label = path_label.split(
                        self.cfg.DATA.PATH_LABEL_SEPARATOR
                    )
                    for idx in range(self._num_clips):
                        self._path_to_videos.append(
                            os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                        )
                        self._labels.append(int(label))
                        self._spatial_temporal_idx.append(idx)
                        self._video_meta[clip_idx * self._num_clips + idx] = {}
            assert (len(self._path_to_videos) > 0), f"Failed to load UCF101 split {self._split_idx} from {path_to_file}"
            print(f"Constructing UCF101 dataloader (size: {len(self._path_to_videos)}) from {path_to_file}")



    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        #short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        fname = self._path_to_videos[index]
        #
        try:
            vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
        except:
            print(" video cannot be loaded by decord : {}".format(fname))
            return []
        #
        