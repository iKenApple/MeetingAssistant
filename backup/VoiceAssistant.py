import json
import re
import threading
import os
import pyaudio
import wave
import time
import subprocess
import numpy as np
import ast
import sys
import pygame
import requests
import pyttsx3
from gtts import gTTS
from cozepy import COZE_CN_BASE_URL, Coze, TokenAuth
from VoiceprintRecognizer import VoiceprintRecognizer


# 配置类
class Config:
    # 音频参数
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 2048
    SILENCE_DURATION = 2
    THRESHOLD = 300
    MAX_DURATION = 300

    # 路径配置
    RECORDINGS_DIR = "../recordings"
    SUMMARY_FILE = "summary.mp3"
    AI_RESPONSE_FILE = "ai.mp3"

    # Coze配置
    API_TOKEN = 'pat_KOHb9a6TPJWs504SZYdrKdJNiG2a1JLZro9rtSkQHnzHkDl7ViPEBPa64Tiiovjg'
    BASE_URL = COZE_CN_BASE_URL
    WORKFLOW_IDS = {
        'record': '7478326457301008394',
        'guest': '7480057361940725795',
        'summary': '7478981906383781942'
    }


class AudioRecorder:
    def __init__(self):
        self._audio = pyaudio.PyAudio()
        self.frames = []
        self.recording = False
        self.last_active = time.time()
        self.has_voice = False

    def __enter__(self):
        self.stream = self._audio.open(
            format=Config.FORMAT,
            channels=Config.CHANNELS,
            rate=Config.RATE,
            input=True,
            frames_per_buffer=Config.CHUNK
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stream.stop_stream()
        self.stream.close()
        self._audio.terminate()

    def record(self):
        try:
            while True:
                data = self.stream.read(Config.CHUNK, exception_on_overflow=False)
                rms = self._calculate_rms(data)

                if rms > Config.THRESHOLD:
                    if not self.recording:
                        print("\n🎙️ 检测到语音，开始录音...")
                        self.recording = True
                        self.frames = []
                    self.last_active = time.time()
                    self.has_voice = True
                    self.frames.append(data)
                elif self.recording:
                    self.frames.append(data)
                    current_time = time.time()

                    if current_time - self.last_active > Config.SILENCE_DURATION:
                        print("\n🔇 持续静音，停止录音")
                        break

                    if current_time - self.last_active > Config.MAX_DURATION:
                        print("\n⏰ 已达最长录音时长")
                        break

        except KeyboardInterrupt:
            print("\n⏹️ 用户中断录音")
            return None

        if not self.has_voice:
            print("\n❌ 未检测到有效语音")
            return None

        return self._save_recording()

    def _calculate_rms(self, data):
        audio_data = np.frombuffer(data, dtype=np.int16)
        if audio_data.size == 0:
            return 0.0
        return np.sqrt(np.mean(audio_data.astype(np.float64) ** 2))

    def _save_recording(self):
        os.makedirs(Config.RECORDINGS_DIR, exist_ok=True)
        filename = os.path.join(Config.RECORDINGS_DIR, f"recording_{int(time.time())}.wav")
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(Config.CHANNELS)
            wf.setsampwidth(self._audio.get_sample_size(Config.FORMAT))
            wf.setframerate(Config.RATE)
            wf.writeframes(b''.join(self.frames))
        return filename


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


class MeetingManager:
    def __init__(self):
        self.vr = VoiceprintRecognizer()
        self.coze = CozeClient()
        self._init_audio_system()

    def _init_audio_system(self):
        pygame.mixer.init()
        self.pp = pyttsx3.init()
        self._check_microphone()

    def _check_microphone(self):
        try:
            audio = pyaudio.PyAudio()
            if audio.get_device_count() == 0:
                raise RuntimeError("未检测到音频输入设备")

            stream = audio.open(
                format=Config.FORMAT,
                channels=Config.CHANNELS,
                rate=Config.RATE,
                input=True,
                frames_per_buffer=Config.CHUNK,
                start=False
            )

            try:
                stream.start_stream()
                for _ in range(3):
                    data = stream.read(Config.CHUNK, exception_on_overflow=False)
                    if len(data) == 0:
                        raise RuntimeError("无法从麦克风读取数据")
                    if self._calculate_rms(data) > 1000:
                        raise RuntimeError("检测到强烈背景噪音，请检查环境")
            finally:
                stream.stop_stream()
                stream.close()
                audio.terminate()

        except Exception as e:
            print(f"麦克风检查失败: {str(e)}")
            print("请检查：1.麦克风权限 2.设备连接 3.背景噪音")
            sys.exit(1)

    @staticmethod
    def _calculate_rms(data):
        return np.sqrt(np.mean(np.frombuffer(data, dtype=np.int16).astype(np.float64) ** 2))

    def _parallel_process(self, filename):
        result = {'speaker': None, 'score': None, 'text': None}

        def speech_to_text():
            try:
                cmd = f"wenet --language chinese {filename}"
                output = subprocess.run(
                    cmd,
                    shell=True,
                    text=True,
                    capture_output=True,
                    timeout=15
                )
                if dict_match := re.search(r"\{.*?\}", output.stdout.strip()):
                    result['text'] = ast.literal_eval(dict_match.group()).get("text")
            except Exception as e:
                print(f"语音识别失败: {str(e)}")

        def identify_speaker():
            try:
                speaker, score = self.vr.identify_speaker(filename)
                result.update({'speaker': speaker, 'score': score})
            except Exception as e:
                print(f"声纹识别失败: {str(e)}")

        threads = [
            threading.Thread(target=speech_to_text),
            threading.Thread(target=identify_speaker)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        return result

    def _process_ai_response(self, content):
        result = {'output': None, 'url': None, 'query': None}

        # 会议总结处理
        summary_data = self.coze.run_workflow('summary', content)
        if summary_data.get("query") == 1:
            self._handle_summary(summary_data, result)

        # AI响应处理
        guest_data = self.coze.run_workflow('guest', content)
        self._handle_guest_response(guest_data, result)

    def _handle_summary(self, data, result):
        result['output'] = f'小U: {data["output"]}'
        print(f"{result['output']}\n{'=' * 50}")
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
            print(f"{result['output']}\n{'=' * 50}")
            response = requests.get(result["url"])
            with open(os.path.join(Config.RECORDINGS_DIR, Config.AI_RESPONSE_FILE), 'wb') as f:
                f.write(response.content)
            AudioPlayer.play(os.path.join(Config.RECORDINGS_DIR, Config.AI_RESPONSE_FILE))

    def run(self):
        while True:
            try:
                with AudioRecorder() as recorder:
                    if filename := recorder.record():
                        start_time = time.time()

                        # 并行处理识别任务
                        result = self._parallel_process(filename)
                        print(f"识别+转文字耗时: {time.time() - start_time:.2f}秒")

                        if result.get('text'):
                            print(f"\n📝 识别结果:\n{'=' * 50}")
                            print(f"{result['speaker']}({result['score']}): {result['text']}\n{'=' * 50}")

                            # 异步保存会议记录
                            content = f"{result['speaker']}({result['score']}): {result['text']}"
                            threading.Thread(
                                target=self.coze.run_workflow,
                                args=('record', content)
                            ).start()

                            # 处理AI响应
                            self._process_ai_response(content)

                        print(f"单次全流程耗时: {time.time() - start_time:.2f}秒")

            except Exception as e:
                print(f"运行错误: {str(e)}")
                time.sleep(1)


if __name__ == "__main__":
    MeetingManager().run()