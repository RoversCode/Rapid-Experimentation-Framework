#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   processor.py
@Time    :   2025/01/14 14:23:45
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   数据管道
"""

# here put the import lib
import logging
import torchaudio


logger = logging.getLogger(__name__)


def resample(data, resample_rate=22050, min_sample_rate=16000, mode="train"):
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
            assert "sample_rate" in sample
            assert "speech" in sample
            sample_rate = sample["sample_rate"]
            waveform = sample["speech"]
            if sample_rate != resample_rate:
                if sample_rate < min_sample_rate:
                    continue
                sample["sample_rate"] = resample_rate
                sample["speech"] = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=resample_rate
                )(waveform)
            max_val = sample["speech"].abs().max()
            if max_val > 1:  # 归一化
                sample["speech"] /= max_val
            yield sample
        except Exception as ex:
            logger.error("Failed to resample {}, ex info {}".format(sample["utt"], ex))
