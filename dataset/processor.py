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
            if sample_rate != data_conf.target_sample_rate:
                sample["sr"] = data_conf.target_sample_rate
                sample["audio_data"] = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=data_conf.target_sample_rate
                )(audio_data)
            max_val = sample["speech"].abs().max()
            if max_val > 1:  # 归一化
                sample["speech"] /= max_val
            yield sample
        except Exception as ex:
            logging.error("Failed to resample {}, ex info {}".format(sample["utt"], ex))


# 第五道流水线 ->  hifigan的mel_spectrogram处理
def compute_fbank(data, feat_extractor, version='cosyvoice1', mode="train"):
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
    for sample in data:
        try:
            waveform = sample["speech"]
            mat = (
                feat_extractor(waveform).squeeze(dim=0).transpose(0, 1)
            )  # [b, num_mels, t] - > [t, num_mels]
            if version == 'cosyvoice2': # @ljj: 强制保证mel frame和speech_token的倍数是两倍
                token_len = min(int(mat.size(0)/2), len(sample['speech_token']))
                mat = mat[:2*token_len, :]
                sample['speech_token'] = sample['speech_token'][:token_len]
            del sample["speech"]
            sample["speech_feat"] = mat  # speech_feat -> mel
            yield sample
        except Exception as ex:
            logging.error(
                "Failed to compute fbank {}, ex info {}".format(sample["utt"], ex)
            )

# @ljj: 有可能是flow部分需要的
def compute_f0(data, pitch_extractor, mode='train'):
    """ Extract f0

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'speech' in sample
        assert 'utt' in sample
        assert 'text_token' in sample
        waveform = sample['speech']
        mat = pitch_extractor(waveform).transpose(1, 2)
        mat = F.interpolate(mat, size=sample['speech_feat'].shape[0], mode='linear')
        sample['pitch_feat'] = mat[0, 0]
        yield sample


def truncate(data, truncate_length=24576, mode='train'):
    """ Truncate data.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            truncate_length: truncate length

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        waveform = sample['speech']
        if waveform.shape[1] > truncate_length:
            start = random.randint(0, waveform.shape[1] - truncate_length)
            waveform = waveform[:, start: start + truncate_length]
        else:
            waveform = torch.concat([waveform, torch.zeros(1, truncate_length - waveform.shape[1])], dim=1)
        sample['speech'] = waveform
        yield sample


# 第六道流水线 -> speaker_embedding 转换成张量并归一化
def parse_embedding(data, normalize, use_spk_embedding=False, mode="train"):
    """Parse utt_embedding/spk_embedding

    Args:
        data: Iterable[{key, wav, label, sample_rate}]

    Returns:
        Iterable[{key, feat, label}]
    """
    for sample in data:
        try:
            if use_spk_embedding:
                sample["embedding"] = torch.tensor(
                    sample["spk_embedding"], dtype=torch.float32
                )
                if normalize:
                    sample["embedding"] = F.normalize(sample["embedding"], dim=0)
            else:
                sample["embedding"] = torch.tensor(
                    sample["utt_embedding"], dtype=torch.float32
                )
                if normalize:
                    sample["embedding"] = F.normalize(sample["embedding"], dim=0)
            yield sample
        except Exception as ex:
            logging.error(
                "Failed to parse embedding {}, ex info {}".format(sample["utt"], ex)
            )


# 第七道流水线 -> 数据"蓄水池"，乱序。
def shuffle(data, shuffle_size=10000, mode="train"):
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
            if len(buf) >= shuffle_size:
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
def sort(data, sort_size=500, mode="train"):
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
    buf.sort(key=lambda x: x["speech_feat"].size(0))
    for x in buf:
        yield x


# 第九道流水线
def batch(
    data, batch_type="static", batch_size=16, max_frames_in_batch=12000, mode="train"
):
    """Wrapper for static/dynamic batch

    配置文件默认 ->  batch_type: dynamic, max_frames_in_batch: 2000
    """
    if mode == "inference":
        return static_batch(data, 1)
    else:
        if batch_type == "static":
            return static_batch(data, batch_size)
        elif batch_type == "dynamic":
            return dynamic_batch(data, max_frames_in_batch)
        else:
            logging.fatal("Unsupported batch type {}".format(batch_type))


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
            assert "speech_feat" in sample
            assert isinstance(sample["speech_feat"], torch.Tensor)
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
def padding(data, use_spk_embedding=False, mode="train"):
    """Padding the data into training data

    Args:
        data: Iterable[List[{key, feat, label}]]

    Returns:
        Iterable[Tuple(keys, feats, labels, feats lengths, label lengths)]
    """
    for sample in data:
        try:
            assert isinstance(sample, list)
            speech_feat_len = torch.tensor(
                [x["speech_feat"].size(1) for x in sample], dtype=torch.int32
            )  # 每个样本的mel长度
            order = torch.argsort(speech_feat_len, descending=True)  # 降序

            utts = [sample[i]["utt"] for i in order]
            speech_token = [torch.tensor(sample[i]["speech_token"]) for i in order]
            speech_token_len = torch.tensor(
                [i.size(0) for i in speech_token], dtype=torch.int32
            )  # speech token长度
            # 填充
            speech_token = pad_sequence(speech_token, batch_first=True, padding_value=0)
            speech_feat = [sample[i]["speech_feat"] for i in order]  # [t, num_mels]
            speech_feat_len = torch.tensor(
                [i.size(0) for i in speech_feat], dtype=torch.int32
            )  #
            speech_feat = pad_sequence(speech_feat, batch_first=True, padding_value=0)
            text = [sample[i]["text"] for i in order]
            text_token = [torch.tensor(sample[i]["text_token"]) for i in order]
            text_token_len = torch.tensor(
                [i.size(0) for i in text_token], dtype=torch.int32
            )
            text_token = pad_sequence(text_token, batch_first=True, padding_value=0)
            # embedding = torch.stack(
            #     [sample[i]["embedding"] for i in order], dim=0
            # )
            batch = {
                "utts": utts,
                "speech_token": speech_token,
                "speech_token_len": speech_token_len,
                "speech_feat": speech_feat,
                "speech_feat_len": speech_feat_len,
                "text": text,
                "text_token": text_token,
                "text_token_len": text_token_len,
                # "embedding": embedding,
            }
            if mode == "inference":
                tts_text = [sample[i]["tts_text"] for i in order]
                tts_index = [sample[i]["tts_index"] for i in order]
                tts_text_token = [
                    torch.tensor(sample[i]["tts_text_token"]) for i in order
                ]
                tts_text_token_len = torch.tensor(
                    [i.size(0) for i in tts_text_token], dtype=torch.int32
                )
                tts_text_token = pad_sequence(
                    tts_text_token, batch_first=True, padding_value=-1
                )
                batch.update(
                    {
                        "tts_text": tts_text,
                        "tts_index": tts_index,
                        "tts_text_token": tts_text_token,
                        "tts_text_token_len": tts_text_token_len,
                    }
                )
            # if use_spk_embedding is True:
            #     batch["embedding"] = batch["spk_embedding"]
            # else:
            #     batch["embedding"] = batch["utt_embedding"]
            yield batch
        except Exception as ex:
            logging.error("Failed to padding {}, ex info {}".format(sample, ex))
