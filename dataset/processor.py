# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
import json
import pyarrow.parquet as pq
import torch
import torchaudio
import torch.nn.functional as F
from io import BytesIO
from torch.nn.utils.rnn import pad_sequence

torchaudio.set_audio_backend("soundfile")


# 第三道流水线
def filter(
    data,
    data_conf,
    mode="train",
):
    """Filter sample according to feature and label length"""
    for sample in data:
        try:
            audio_data = sample["audio_data"]
            sr = sample["sr"]
            dur = len(audio_data) / sr
            if dur > data_conf.max_duration:
                audio_data = audio_data[: int(data_conf.max_duration * sr)]
                sample["audio_data"] = audio_data
            yield sample
        except Exception as ex:
            logging.error("Failed to filter {}, ex info {}".format(sample["utt"], ex))


# 第四道流水线 -> 采样率对齐22050
def resample(data, data_conf, mode="train"):
    """Resample data.
    Inplace operation.

    Args:
        data: Iterable[{key, wav, label, sample_rate}]
        resample_rate: target resample rate

    Returns:
        Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        try:
            sample_rate = sample["sr"]
            audio_data = sample["audio_data"]
            # 将numpy数组转换为torch tensor
            audio_tensor = torch.from_numpy(audio_data).float()
            # 确保tensor是二维的 [1, T]
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            if sample_rate != data_conf.sampling_rate:
                sample["sr"] = data_conf.sampling_rate
                # 重采样
                audio_tensor = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=data_conf.sampling_rate
                )(audio_tensor)
            max_val = audio_tensor.abs().max()
            if max_val > 1:  # 归一化
                audio_tensor /= max_val
            sample["audio_data"] = audio_tensor
            yield sample
        except Exception as ex:
            logging.error("Failed to resample {}, ex info {}".format(sample["utt"], ex))


# 第五道流水线 ->  hifigan的mel_spectrogram处理
def compute_fbank(data, data_conf, mode="train"):
    """Extract fbank
    频谱参数
        n_fft: 1024
        num_mels: 80
        sampling_rate: !ref <sample_rate>
        hop_size: 256
        win_size: 1024
        fmin: 0
        fmax: 8000
        center: False
    Args:
        data: Iterable[{key, wav, label, sample_rate}]

    Returns:
        Iterable[{key, feat, label}]
    """
    # from utils.audio import mel_spectrogram
    # feat_extractor = mel_spectrogram(
    #     n_fft=data_conf.n_fft,
    #     num_mels=data_conf.num_mels,
    #     sampling_rate=data_conf.sampling_rate,
    #     hop_size=data_conf.hop_size,
    #     win_size=data_conf.win_size,
    # )
    from utils.wav_frontend import WavFrontend
    frontend = WavFrontend(**data_conf.frontend_conf)
    for sample in data:
        try:
            waveform = sample["audio_data"]
            lengths = [waveform.shape[1]]
            feat, feats_lens = frontend(waveform, lengths)
            mat = feat.squeeze(dim=0)  # [t, num_mels]
            del sample["audio_data"]
            sample["speech_feat"] = mat
            yield sample
        except Exception as ex:
            logging.error(
                "Failed to compute fbank {}, ex info {}".format(sample["utt"], ex)
            )


# 第七道流水线 -> 数据"蓄水池"，乱序。
def shuffle(data, data_conf, mode="train"):
    """Local shuffle the data
      数据"蓄水池"，乱序。
    Args:
        data: Iterable[{key, feat, label}]
        shuffle_size: buffer size for shuffle

    Returns:
        Iterable[{key, feat, label}]
    """
    buf = []
    for sample in data:
        try:
            buf.append(sample)
            if len(buf) >= data_conf.buffer_size:
                random.shuffle(buf)
                for x in buf:
                    yield x
                buf = []
        except Exception as ex:
            logging.error("Failed to shuffle {}, ex info {}".format(sample["utt"], ex))
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


# 第八道流水线 -> 长度差不多的数据放在一起
def sort(data, data_conf, sort_size=50, mode="train"):
    """Sort the data by feature length.
    Sort is used after shuffle and before batch, so we can group
    utts with similar lengths into a batch, and `sort_size` should
    be less than `shuffle_size`

    Args:
        data: Iterable[{key, feat, label}]
        sort_size: buffer size for sort

    Returns:
        Iterable[{key, feat, label}]
    """

    buf = []
    for sample in data:
        try:
            buf.append(sample)
            if len(buf) >= sort_size:
                buf.sort(key=lambda x: x["speech_feat"].size(0))
                for x in buf:
                    yield x
                buf = []
        except Exception as ex:
            logging.error("Failed to sort {}, ex info {}".format(sample["utt"], ex))
    # The sample left over
    buf.sort(key=lambda x: x["mel"].size(0))
    for x in buf:
        yield x


# 第九道流水线
def batch(data, data_conf, mode="train"):
    """Wrapper for static/dynamic batch

    配置文件默认 ->  batch_type: dynamic, max_frames_in_batch: 2000
    """
    if mode == "inference":
        return static_batch(data, 1)
    else:
        if data_conf.batch_type == "static":
            return static_batch(data, data_conf.batch_size)
        elif data_conf.batch_type == "dynamic":
            return dynamic_batch(data, data_conf.max_frames_in_batch)
        else:
            logging.fatal("Unsupported batch type {}".format(data_conf.batch_type))


def static_batch(data, batch_size=16):
    """Static batch the data by `batch_size`

    Args:
        data: Iterable[{key, feat, label}]
        batch_size: batch size

    Returns:
        Iterable[List[{key, feat, label}]]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= batch_size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def dynamic_batch(data, max_frames_in_batch=12000, mode="train"):
    """Dynamic batch the data until the total frames in batch
    reach `max_frames_in_batch`

    Args:
        data: Iterable[{key, feat, label}]
        max_frames_in_batch: max_frames in one batch

    Returns:
        Iterable[List[{key, feat, label}]]
    """
    buf = []
    longest_frames = 0
    for sample in data:
        try:
            new_sample_frames = sample["speech_feat"].size(0)
            longest_frames = max(longest_frames, new_sample_frames)
            frames_after_padding = longest_frames * (
                len(buf) + 1
            )  # 乘以batch_size表示当前batch的总帧数
            if frames_after_padding > max_frames_in_batch:
                yield buf
                buf = [sample]
                longest_frames = new_sample_frames
            else:
                buf.append(sample)
        except Exception as ex:
            logging.error(
                "Failed to dynamic batch {}, ex info {}".format(sample["utt"], ex)
            )
    if len(buf) > 0:
        yield buf


# 第十道流水线 -> 填充长度
def padding(data, data_conf, mode="train"):
    """Padding the data into training data

    Args:
        data: Iterable[List[{key, feat, label}]]

    Returns:
        Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
    for sample in data:
        try:
            speech_feats = [x["speech_feat"] for x in sample]
            labels = [int(x["label"]) for x in sample]
            speech_feats_lens = [x["speech_feat"].size(0) for x in sample]
            speech_feats_pad = pad_sequence(speech_feats, batch_first=True, padding_value=0.0)
            speech_feats_lens = torch.tensor(speech_feats_lens, dtype=torch.int32)
            
            batch = {
                "speech_feat": speech_feats_pad,
                "speech_feat_len": speech_feats_lens,
                "labels": labels
            }
            yield batch
        except Exception as ex:
            logging.error("Failed to padding {}, ex info {}".format(sample, ex))
