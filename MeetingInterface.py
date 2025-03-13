#!/usr/bin/env python3

import argparse
import json
import os
import sys
import threading
from collections import defaultdict
import numpy as np
import pygame
import pyttsx3
import requests
from colorama import Fore, Style, init
from cozepy import COZE_CN_BASE_URL, Coze, TokenAuth

try:
    import sounddevice as sd
except ImportError:
    print("请安装sounddevice: pip install sounddevice")
    sys.exit(-1)

import sherpa_onnx
import soundfile as sf

# 全局音频参数
SAMPLE_RATE = 16000  # 统一采样率
SAMPLES_PER_READ = int(0.1 * SAMPLE_RATE)  # 100ms每次读取

class Config:
    # 路径配置
    RECORDINGS_DIR = "./recordings"
    AI_RESPONSE_FILE = "ai.mp3"

    # Coze配置
    API_TOKEN = 'pat_KOHb9a6TPJWs504SZYdrKdJNiG2a1JLZro9rtSkQHnzHkDl7ViPEBPa64Tiiovjg'
    BASE_URL = COZE_CN_BASE_URL
    WORKFLOW_IDS = {
        'record': '7478326457301008394',
        'guest': '7480057361940725795',
        'summary': '7478981906383781942'
    }

class CozeClient:
    def __init__(self):
        self.client = Coze(
            auth=TokenAuth(token=Config.API_TOKEN),
            base_url=Config.BASE_URL
        )

    def run_workflow(self, workflow_type, content):
        workflow_id = Config.WORKFLOW_IDS[workflow_type]
        response = self.client.workflows.runs.create(
            workflow_id=workflow_id,
            parameters={"input": content}
        )
        return json.loads(response.data)

class AudioPlayer:
    @staticmethod
    def play(file_path):
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            print(f"音频播放失败: {str(e)}")

class MeetingAssistant:
    def __init__(self):
        self.coze = CozeClient()
        self._init_audio_system()
        init(autoreset=True)  # 自动重置颜色

    def _init_audio_system(self):
        pygame.mixer.init()
        self.pp = pyttsx3.init()

    def _process_ai_response(self, content):
        result = {'output': None, 'url': None, 'query': None}

        print("会议总结工作流处理中...")
        # 会议总结处理
        summary_data = self.coze.run_workflow('summary', content)
        if summary_data.get("query") == 1:
            self._handle_summary(summary_data, result)
        print("会议总结工作流处理结束")

        print("会议嘉宾工作流处理中...")
        # AI响应处理
        guest_data = self.coze.run_workflow('guest', content)
        self._handle_guest_response(guest_data, result)
        print("会议嘉宾工作流处理结束")

    def _handle_summary(self, data, result):
        result['output'] = f'小U: {data["output"]}'
        print(f"{result['output']}\n")
        self.pp.say(result['output'])
        self.pp.runAndWait()
        #tts = gTTS(text=result['output'], lang='zh-cn')
        #tts.save(os.path.join(Config.RECORDINGS_DIR, Config.SUMMARY_FILE))
        #AudioPlayer.play(os.path.join(Config.RECORDINGS_DIR, Config.SUMMARY_FILE))

    def _handle_guest_response(self, data, result):
        result.update({
            'output': f'小U: {data["output"]}',
            'url': data["url"]
        })
        if '小U收到' not in data["output"]:
            print(f"{result['output']}\n")
            response = requests.get(result["url"])
            with open(os.path.join(Config.RECORDINGS_DIR, Config.AI_RESPONSE_FILE), 'wb') as f:
                f.write(response.content)
            AudioPlayer.play(os.path.join(Config.RECORDINGS_DIR, Config.AI_RESPONSE_FILE))

def get_merged_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # 语音识别参数
    parser.add_argument("--tokens", required=True, help="tokens.txt路径")
    parser.add_argument("--encoder", required=True, help="编码器模型路径")
    parser.add_argument("--decoder", required=True, help="解码器模型路径")
    parser.add_argument("--joiner", required=True, help="joiner模型路径")

    # 说话人识别参数
    parser.add_argument("--speaker-file", required=True, help="说话人注册文件路径")
    parser.add_argument("--speaker-model", required=True, help="说话人特征提取模型路径")
    parser.add_argument("--silero-vad-model", required=True, help="VAD模型路径")

    # 通用参数
    parser.add_argument("--threshold", type=float, default=0.6, help="说话人识别阈值")
    parser.add_argument("--num-threads", type=int, default=4, help="计算线程数")
    parser.add_argument("--provider", default="cpu", help="cpu/cuda/coreml")
    parser.add_argument("--decoding-method", default="greedy_search",
                        help="greedy_search/modified_beam_search")

    return parser.parse_args()


def init_speech_recognizer(args):
    """初始化语音识别模型"""
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        num_threads=args.num_threads,
        sample_rate=SAMPLE_RATE,
        feature_dim=80,
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=300,
        decoding_method=args.decoding_method,
        provider=args.provider,
    )
    return recognizer


def init_speaker_system(args):
    """初始化说话人识别系统"""
    # 初始化特征提取器
    speaker_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=args.speaker_model,
        num_threads=args.num_threads,
        provider=args.provider
    )
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(speaker_config)

    # 初始化VAD
    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = args.silero_vad_model
    vad_config.silero_vad.min_silence_duration = 0.5
    vad_config.silero_vad.min_speech_duration = 0.5
    vad_config.sample_rate = SAMPLE_RATE
    vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=100)

    return extractor, vad


def load_speaker_profiles(args, extractor):
    """加载预注册的说话人声纹特征"""
    speaker_db = defaultdict(list)
    with open(args.speaker_file) as f:
        for line in f:
            if not line.strip(): continue
            name, path = line.strip().split()
            speaker_db[name].append(path)

    manager = sherpa_onnx.SpeakerEmbeddingManager(extractor.dim)
    for name, paths in speaker_db.items():
        embeddings = []
        for path in paths:
            audio, _ = sf.read(path, dtype="float32", always_2d=True)
            audio = audio[:, 0]  # 单声道

            stream = extractor.create_stream()
            stream.accept_waveform(SAMPLE_RATE, audio)
            stream.input_finished()
            embeddings.append(np.array(extractor.compute(stream)))

        avg_embedding = np.mean(embeddings, axis=0)
        manager.add(name, avg_embedding)

    return manager


def audio_capture_loop(args, assistant):
    """主处理循环"""
    # 初始化模型
    speech_recognizer = init_speech_recognizer(args)
    speaker_extractor, vad = init_speaker_system(args)
    speaker_manager = load_speaker_profiles(args, speaker_extractor)

    # 创建语音识别流
    asr_stream = speech_recognizer.create_stream()

    # 音频缓冲区
    audio_buffer = np.array([], dtype=np.float32)

    print("会议记录员已准备就绪，请开始圆桌会议：")
    with sd.InputStream(channels=1, dtype=np.float32, samplerate=SAMPLE_RATE) as mic:
        while True:
            # 读取麦克风数据
            samples, _ = mic.read(SAMPLES_PER_READ)
            samples = samples.ravel()

            # 同时发送到语音识别和VAD
            asr_stream.accept_waveform(SAMPLE_RATE, samples)

            # VAD处理
            audio_buffer = np.concatenate([audio_buffer, samples])
            while len(audio_buffer) >= vad.config.silero_vad.window_size:
                vad.accept_waveform(audio_buffer[:vad.config.silero_vad.window_size])
                audio_buffer = audio_buffer[vad.config.silero_vad.window_size:]

            # 实时语音识别
            while speech_recognizer.is_ready(asr_stream):
                speech_recognizer.decode_stream(asr_stream)

            # 获取当前识别结果
            partial_result = speech_recognizer.get_result(asr_stream)

            # 处理完整语音段
            while not vad.empty():
                speech_segment = vad.front.samples
                vad.pop()

                # 说话人识别
                speaker_stream = speaker_extractor.create_stream()
                speaker_stream.accept_waveform(SAMPLE_RATE, speech_segment)
                speaker_stream.input_finished()
                embedding = np.array(speaker_extractor.compute(speaker_stream))
                speaker = speaker_manager.search(embedding, args.threshold) or "听众"

                # 获取最终识别结果
                final_text = speech_recognizer.get_result(asr_stream)
                content = f"\n{speaker}: {final_text}"
                print(Fore.BLUE + Style.BRIGHT + content)
                speech_recognizer.reset(asr_stream)
                partial_result = None

                # 异步保存会议记录
                threading.Thread(
                    target=assistant.coze.run_workflow,
                    args=('record', content)
                ).start()

                assistant._process_ai_response(content)

            # 显示实时结果
            if partial_result:
                sys.stdout.write(Fore.GREEN + Style.BRIGHT + f"\r实时转录: {partial_result}")
                sys.stdout.flush()


if __name__ == "__main__":
    try:
        args = get_merged_args()
        audio_capture_loop(args, MeetingAssistant())
    except KeyboardInterrupt:
        print("\n程序已终止")