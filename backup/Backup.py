import json
import re
import threading

import pyaudio
import wave
import time
import subprocess
import numpy as np
import ast
import sys

import pygame
import requests
from cozepy import COZE_CN_BASE_URL
from cozepy import Coze, TokenAuth, Message, ChatStatus, MessageContentType  # noqa

from backup.VoiceprintRecognizer import VoiceprintRecognizer

from gtts import gTTS

# 配置参数
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 2048
SILENCE_DURATION = 2
THRESHOLD = 300
MAX_DURATION = 300

# Get an access_token through personal access token or oauth.
coze_api_token = 'pat_KOHb9a6TPJWs504SZYdrKdJNiG2a1JLZro9rtSkQHnzHkDl7ViPEBPa64Tiiovjg'
# The default access is api.coze.com, but if you need to access api.coze.cn,
# please use base_url to configure the api endpoint to access
coze_api_base = COZE_CN_BASE_URL

# Init the Coze client through the access_token.
coze = Coze(auth=TokenAuth(token=coze_api_token), base_url=coze_api_base)

# Create a workflow instance in Coze, copy the last number from the web link as the workflow's ID.
# 会议记录工作流
workflow_id_meeting_record = '7478326457301008394'
workflow_id_meeting_guest = '7480057361940725795'
workflow_id_meeting_summary = '7478981906383781942'

# 初始化pygame
pygame.mixer.init()

def check_microphone():
    """检查麦克风是否正常工作"""
    audio = pyaudio.PyAudio()
    try:
        if audio.get_device_count() == 0:
            raise Exception("未检测到音频输入设备")

        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True,
                            frames_per_buffer=CHUNK,
                            start=False)

        try:
            stream.start_stream()
            for _ in range(3):  # 尝试读取三次
                data = stream.read(CHUNK, exception_on_overflow=False)
                if len(data) == 0:
                    raise Exception("无法从麦克风读取数据")
                if calculate_rms(data) > 1000:  # 检测明显环境噪音
                    raise Exception("检测到强烈背景噪音，请检查环境")
        finally:
            stream.stop_stream()
            stream.close()

    except Exception as e:
        print(f"麦克风检查失败: {str(e)}")
        print("请检查：1.麦克风权限 2.设备连接 3.背景噪音")
        sys.exit(1)
    finally:
        audio.terminate()

def calculate_rms(data):
    """安全计算音频能量值"""
    audio_data = np.frombuffer(data, dtype=np.int16)
    if audio_data.size == 0:
        return 0.0

    with np.errstate(divide='ignore', invalid='ignore'):
        audio_data = audio_data.astype(np.float64)
        mean_square = np.mean(audio_data ** 2)
        if mean_square <= 0:
            return 0.0
        rms = np.sqrt(mean_square)

    return rms if not np.isnan(rms) else 0.0

def record_audio():
    """执行录音操作"""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    #print("\n🎤 语音检测中...（说「开始」唤醒）")
    frames = []
    recording = False
    last_active = time.time()
    has_voice = False

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            rms = calculate_rms(data)

            # 语音活动检测
            if rms > THRESHOLD:
                if not recording:
                    print("\n🎙️ 检测到语音，开始录音...")
                    recording = True
                    frames = []  # 丢弃之前的静音数据
                last_active = time.time()
                has_voice = True
                frames.append(data)
            elif recording:
                # 在录音状态下处理静音
                frames.append(data)
                if time.time() - last_active > SILENCE_DURATION:
                    print("\n🔇 持续静音，停止录音")
                    break

                if (time.time() - last_active) > MAX_DURATION:
                    print("\n⏰ 已达最长录音时长")
                    break

    except KeyboardInterrupt:
        print("\n⏹️ 用户中断录音")
        return None
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

    if not has_voice:
        print("\n❌ 未检测到有效语音")
        return None

    # 保存录音文件
    filename = f"recordings/recording_{int(time.time())}.wav"
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    return filename

def speech_to_text(filename, result):
    """调用语音识别引擎"""
    try:
        print(f"\n🔍 正在识别: {filename}")
        cmd = f"wenet --language chinese {filename}"
        output = subprocess.run(
            cmd,
            shell=True,
            text=True,
            capture_output=True,
            timeout=15
        )
        output_str = output.stdout.strip()

        # 清理输出（如果包含非字典内容）
        dict_match = re.search(r"\{.*?\}", output_str)
        if not dict_match:
            raise ValueError("输出中未找到有效字典")

        output_dict = ast.literal_eval(dict_match.group())
        result['text'] = output_dict.get("text")
    except subprocess.TimeoutExpired:
        print("识别超时，请检查wenet配置")
        result['text'] = None
    except Exception as e:
        print(f"识别失败: {str(e)}")
        result['text'] = None

def identify_speaker(filename, vr, result):
    """调用声纹识别引擎"""
    try:
        speaker, score = vr.identify_speaker(filename)
        result['speaker'] = speaker
        result['score'] = score
    except Exception as e:
        print(f"声纹识别失败: {str(e)}")
        result['speaker'] = None
        result['score'] = None

def save_meeting_record(content, result_by_ai):
    # 发送给coze工作流，做会议记录
    coze.workflows.runs.create(
        workflow_id=workflow_id_meeting_record,
        parameters={"input": content}
    )

def meeting_summary(content, result_by_ai):
    # 发送给coze工作流，做会议记录
    workflow = coze.workflows.runs.create(
        workflow_id=workflow_id_meeting_summary,
        parameters={"input": content}
    )
    parsed_data = json.loads(workflow.data)
    result_by_ai['output'] = f'小U: {parsed_data["output"]}'
    result_by_ai['query'] = parsed_data['query']
    result_by_ai['url'] = parsed_data["url"]

    print(parsed_data)
    if result_by_ai['query'] == 1:
        print("meeting_summary:")
        print(result_by_ai['output'])
        print("=" * 50)
        tts = gTTS(text=result_by_ai['output'], lang='zh-cn')
        # 将文本转为语音并保存为音频文件
        tts.save("recordings/summary.mp3")
        # 加载MP3文件
        pygame.mixer.music.load("recordings/summary.mp3")

        # 播放MP3文件
        pygame.mixer.music.play()

        # 等待音乐播放完毕
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)


def get_ai_response(content, result_by_ai):
    # 发送给coze工作流，嘉宾发言
    workflow = coze.workflows.runs.create(
        workflow_id=workflow_id_meeting_guest,
        parameters={"input": content}
    )
    parsed_data = json.loads(workflow.data)

    result_by_ai['output'] = f'小U: {parsed_data["output"]}'
    result_by_ai['url'] = parsed_data["url"]

    if '小U收到' not in parsed_data["output"]:
        print("get_ai_response:")
        print(result_by_ai['output'])
        print("=" * 50)
        #  播放小U发言
        response = requests.get(result_by_ai["url"])
        with open("recordings/ai.mp3", 'wb') as file:
            file.write(response.content)

        # 加载MP3文件
        pygame.mixer.music.load("recordings/ai.mp3")

        # 播放MP3文件
        pygame.mixer.music.play()

        # 等待音乐播放完毕
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

if __name__ == "__main__":
    vr = VoiceprintRecognizer()

    check_microphone()

    while True:
        try:
            filename = record_audio()
            if filename:
                start_time = time.time()

                # 创建字典存储结果
                result = {'speaker': None, 'score': None, 'text': None}
                result_by_ai = {'output': None, 'url': None, 'query': None}

                # 创建线程
                threads = [
                    threading.Thread(target=identify_speaker, args=(filename, vr, result)),
                    threading.Thread(target=speech_to_text, args=(filename, result)),
                ]
                # 启动线程
                for thread in threads:
                    thread.start()

                # 等待所有线程完成
                for thread in threads:
                    thread.join()

                end_time = time.time()

                # 计算耗时
                execution_time = end_time - start_time
                print(f"识别嘉宾和语音转文字耗时：{execution_time:.6f} 秒")

                # 打印结果
                speaker = result['speaker']
                score = result['score']
                text = result['text']

                if result['speaker'] != None and result['score'] != None:
                    if text:
                        print("\n📝 识别结果:")
                        print("=" * 50)
                        print(f'{speaker}({score}): {text}')
                        print("=" * 50)

                    start_time_tmp = time.time()
                    # 发送给coze工作流，做会议记录
                    #save_meeting_record(f'{speaker}({score}): {text}')
                    content = f'{speaker}({score}): {text}'
                    threading.Thread(target=save_meeting_record, args=(content, result_by_ai)).start()

                    # 会议总结
                    thread_summary = threading.Thread(target=meeting_summary, args=(content, result_by_ai))
                    thread_summary.start()
                    thread_summary.join()

                    end_time = time.time()

                    # 计算耗时
                    execution_time = end_time - start_time_tmp
                    print(f"小U总结耗时：{execution_time:.6f} 秒")

                    # 发送给coze工作流，嘉宾发言
                    thread_response = threading.Thread(target=get_ai_response, args=(content,result_by_ai))
                    thread_response.start()
                    thread_response.join()

                    end_time = time.time()

                    # 计算耗时
                    execution_time = end_time - start_time_tmp
                    print(f"嘉宾发言耗时：{execution_time:.6f} 秒")

                end_time = time.time()

                # 计算耗时
                execution_time = end_time - start_time
                print(f"单次全流程耗时：{execution_time:.6f} 秒")

        except Exception as e:
            print(f"发生未预期错误: {str(e)}")
            time.sleep(1)
            print("重新开始记录...")