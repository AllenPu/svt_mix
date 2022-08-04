import os
import random
import warnings
from decord import VideoReader, cpu
import torch
import torch.utils.data
import numpy as np

from datasets.data_utils import get_random_sampling_rate, tensor_normalize, spatial_sampling, pack_pathway_output
from datasets.decoder import decode
from datasets.video_container import get_video_container
from datasets.transform import VideoDataAugmentationDINO
from einops import rearrange
from decord import VideoReader
from torchvision import transforms
from datasets.decoder import temporal_sampling


class HMDB51(torch.utils.data.Dataset):
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

        print("Constructing HMDB51 {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """

        path_to_file = os.path.join(
                self.cfg.DATA.PATH_TO_DATA_DIR, "hmdb51_{}_split_1_videos.txt".format(self.mode)
            )
  


        assert os.path.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
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
        print(f"Constructing HMDB51 dataloader (size: {len(self._path_to_videos)}) from {path_to_file}")

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
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["val", "test"]:
            temporal_sample_index = (self._spatial_temporal_idx[index] // self.cfg.TEST.NUM_SPATIAL_CROPS)
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                (self._spatial_temporal_idx[index] % self.cfg.TEST.NUM_SPATIAL_CROPS)
                if self.cfg.TEST.NUM_SPATIAL_CROPS > 1 else 1
            )
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3 if self.cfg.TEST.NUM_SPATIAL_CROPS > 1
                else [self.cfg.DATA.TRAIN_JITTER_SCALES[0]] * 2 + [self.cfg.DATA.TEST_CROP_SIZE]
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        sampling_rate = get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatedly find a random video replacement that can be decoded.
        # Decode video. Meta info is used to perform selective decoding.
        #
        # re-implemente decord frams from videos via decord
        #
        try:
            vr = VideoReader(self._path_to_videos[index], num_threads=1, ctx = cpu(0))
        except:
            print("video cannot be loaded by decord: ", self._path_to_videos[index])
            vr = []

        # check if video can be loaded
        assert vr != [], print(" video canno tbe loaded")  
        #
        # we set there are 10 clips per video
        num_clips = 10 
        # uniformly sample the frames to the end
        if self.mode == 'train':
            frames = []
            # 2 global views
            for i in range(2):
                #clip size which is also the index of first clip
                clip_size = int(len(vr)/10)
                # last clip
                end_clip_index = len(vr) - clip_size
                #
                start_idx = random.uniform(0, clip_size)
                #
                end_idx = random.uniform(end_clip_index, len(vr))
                #
                assert start_idx < end_idx, print("the frame is too small!!!")
                #
                index_ = np.linspace(start_idx, end_idx, self.cfg.DATA.NUM_FRAMES)#
                #
                index_ = np.clip(index_, 0, len(vr)).astype(np.int64)
                #
                all_index = list(index_)
                #
                frame = self.convert_to_frame_by_index(vr, all_index)
                # in the val or test
                frames.append(frame)
            # 8 local video clips
            for j in range(8):
                clip_size = int(len(vr)/8)
                # for every clip sample self.cfg.DATA.NUM_FRAMES frames
                # [j, j + len]
                start_idx = random.uniform(j*clip_size, j*clip_size + clip_size)
                # set sperartion [j, j+ len]
                end_idx = start_idx + clip_size
                #
                index_ = np.linspace(start_idx, end_idx, self.cfg.DATA.NUM_FRAMES).astype(np.int64)
                #
                index_ = np.clip(index_, 0, len(vr)-1).astype(np.int64)
                #
                all_index = list(index_)
                #
                frame = self.convert_to_frame_by_index(vr, all_index)
                # in the val or test
                frames.append(frame)
            assert len(frames) == 10, print(" gloabl and local not loaded correctly ! ")
        else:
            # load test or val by the index of clip
            clip_id = temporal_sample_index
            #
            num_clips = 10
            #
            clip_size = int(len(vr)/num_clips)
            #
            start_idx = random.uniform( clip_id*clip_size, (clip_id+1)*clip_size)
            end_idx = start_idx + clip_size - 1
            #
            index = np.linspace(start_idx, end_idx, self.cfg.DATA.NUM_FRAMES)
            #
            index = np.clip(index, 0, len(vr)-1).astype(np.int64)
            #
            all_index = list(index)
            #
            frames = self.convert_to_frame_by_index(vr, all_index)

        label = self._labels[index]


        if self.mode in ["test", "val"]:
            # Perform color normalization.
            frames = tensor_normalize(
                frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                )
            frames = frames.permute(3, 0, 1, 2)

                # Perform data augmentation.
            frames = spatial_sampling(
                frames,
                spatial_idx=spatial_sample_index,
                min_scale=min_scale,
                max_scale=max_scale,
                crop_size=crop_size,
                random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )
                # if not self.cfg.MODEL.ARCH in ['vit']:
                #     frames = pack_pathway_output(self.cfg, frames)
                # else:
                # Perform temporal sampling from the fast pathway.
                # frames = [torch.index_select(
                #     x,
                #     1,
                #     torch.linspace(
                #         0, x.shape[1] - 1, self.cfg.DATA.NUM_FRAMES
                #     ).long(),
                # ) for x in frames]
                # here the frame is at c,t,h,w
                # we need to shape to t,c,h,w
        else: # train
                # T H W C -> T C H W
            frames = [x.permute(0,3,1,2) for x in frames]
                # video aug
            augmentation = VideoDataAugmentationDINO()
                # implment
            frames = augmentation(frames, from_list=True, no_aug=self.cfg.DATA.NO_SPATIAL,
                                      two_token=self.cfg.MODEL.TWO_TOKEN)
            # T C H W -> C T H W
            frames = [x.permute(1,0,2,3) for x in frames]
            # temproal sampling
            frames = [torch.index_select(
                    x,
                    1,
                    torch.linspace(
                        0, x.shape[1] - 1, x.shape[1] if self.cfg.DATA.RAND_FR else self.cfg.DATA.NUM_FRAMES

                    ).long(),
                ) for x in frames]
            
            # train would return a list of len 8 while test would return a list of len 1
        return frames, label, index, {}


    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
        #

    def convert_to_frame_by_index(self, vr, all_index):
        vr.seek(0)
        frame = vr.get_batch(all_index).asnumpy()
        frame = [transforms.ToPILImage()(frame) for frame in frame]
        frame = [transforms.ToTensor()(img) for img in frame]
        frame = torch.stack(frame)
        # to the shape of T H W C
        frame = frame.permute(0, 2, 3, 1)
        # in the val or test
        return frame


