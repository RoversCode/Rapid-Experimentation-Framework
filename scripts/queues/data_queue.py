#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   strong_queue.py
@Time    :   2024/11/11 10:38:20
@Author  :   ChengHee
@Version :   1.0
@Contact :   liujunjie199810@gmail.com
@Desc    :   None
"""

# here put the import lib
import os

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["WANDB_DISABLED"] = "true"
import multiprocessing
import torch
import random
import onnxruntime
import numpy as np
import whisper
import librosa
import time
import threading
from multiprocessing.managers import BaseManager
from pathlib import Path


def data_producer(data_path="Data/models/cosy_abnormal_detection/data_list.txt"):
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split("|")
            samples.append(line)

    print("开始生产数据")
    while True:
        for sample in samples:
            # 读取数据
            try:
                rel_path, label = sample
                path_root = Path("Data/models/cosy_abnormal_detection")
                # 读取音频
                audio_path = path_root / (rel_path + ".mp3")
                if not os.path.exists(audio_path):
                    audio_path = path_root / (rel_path + ".wav")
                if not os.path.exists(audio_path):
                    continue
                audio, sr = librosa.load(audio_path, sr=16000)
                dur = len(audio) / sr
                if dur > 30:
                    audio = audio[: 30 * sr]  # 截断
                producer_queue.put(
                    {
                        "utt": rel_path.split("/")[-1],
                        "audio_data": audio,
                        "sr": sr,
                        "label": label,
                    }
                )
            except Exception as ex:
                print(f"生产数据出错{ex}")
        # 刷新
        print("一轮结束，刷新数据")
        random.shuffle(samples)


def get_speech_token(device_id):
    device_id = str(device_id)

    @torch.no_grad()
    def _get_speech_token():
        # Spk_membedding和Speech和Token的计算
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        option.intra_op_num_threads = 1
        providers = [("CUDAExecutionProvider", {"device_id": int(device_id)})]
        speech_tokenizer_session = onnxruntime.InferenceSession(
            "/datadisk2/liujunjie/growth/audio/CosyVoice/pretrained_models/CosyVoice2-0.5B/speech_tokenizer_v2.onnx",
            sess_options=option,
            providers=providers,
        )
        while True:
            try:
                """
                "utt": rel_path.split('/')[-1],
                "audio_data": audio,
                "sr": sr,
                "label": label,
                """
                sample = producer_queue.get()
                start = time.time()
                feat = whisper.log_mel_spectrogram(sample["audio_data"], n_mels=128)
                if feat.dim() == 2:
                    feat = feat.unsqueeze(0)
                speech_token = speech_tokenizer_session.run(
                    None,
                    {
                        speech_tokenizer_session.get_inputs()[0]
                        .name: feat.detach()
                        .cpu()
                        .numpy(),
                        speech_tokenizer_session.get_inputs()[1].name: np.array(
                            [feat.shape[2]], dtype=np.int32
                        ),
                    },
                )[0].flatten()
                end = time.time()
                print(f"speech token time: {end - start}")
                sample["speech_token"] = speech_token
                # 写成字节流
                sppech_token_queue.put(sample)  # 取的speech token
            except Exception as e:
                print(f"failed speech token: {e}")

    return _get_speech_token


def make_batches(bs=16):
    try:
        cache = list()
        cnt = 0
        while True:
            while cnt < bs:
                sample = sppech_token_queue.get()
                cnt += 1
                cache.append(sample)
            final_queue.put(cache)
            cache = list()
            cnt = 0
    except Exception as e:
        print("make_batches", e)
        cache = list()


# 存入encodec的内容
producer_queue = multiprocessing.Queue(512)
sppech_token_queue = multiprocessing.Queue(512)
final_queue = multiprocessing.Queue(512)


if __name__ == "__main__":

    # 读取音频
    threading.Thread(target=data_producer).start()
    threading.Thread(target=make_batches).start()

    for i in range(6, 8):
        for _ in range(3):
            multiprocessing.Process(target=get_speech_token(i)).start()
            time.sleep(1)

    class QueueManager(BaseManager):
        pass

    QueueManager.register("abnormal_detection_1", callable=lambda: final_queue)
    # 127.0.0.1
    m = QueueManager(address=("192.168.15.87", 8124), authkey=b"liujunjieabracadabra")
    s = m.get_server()
    print("服务启动")
    s.serve_forever()
