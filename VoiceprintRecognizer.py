import json
import os
import wave
import numpy as np
from vosk import Model, SpkModel, KaldiRecognizer


class VoiceprintRecognizer:
    def __init__(self):
        """
        初始化声纹识别系统
        模型路径需要根据实际情况修改
        """
        # 初始化语音识别模型（必需，即使不使用文本）
        self.asr_model = Model("models/vosk-model-cn-0.22")

        # 加载声纹识别模型
        self.spk_model = SpkModel("models/vosk-model-spk-0.4")

        # 注册说话人数据库
        self.speaker_db = {
            "王斯丙": np.load("voice/speaker_sbwang.npy"),
            "吴凡": np.load("voice/fanwu.npy")
        }

        # 相似度阈值（根据实际场景调整）
        self.threshold = 0.5

    def _validate_audio(self, wf):
        """验证音频格式有效性"""
        if wf.getnchannels() != 1:
            raise ValueError("只支持单声道音频")
        if wf.getsampwidth() != 2:
            raise ValueError("只支持16-bit PCM格式")
        if wf.getframerate() != 16000:
            raise ValueError("需要16000Hz采样率")

    def extract_voiceprint(self, wav_path):
        """
        从音频文件中提取声纹特征
        参数：
            wav_path : str - 音频文件路径
        返回：
            np.ndarray - 128维声纹特征向量
        """
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"音频文件不存在：{wav_path}")

        with wave.open(wav_path, "rb") as wf:
            self._validate_audio(wf)

            # 初始化识别器
            rec = KaldiRecognizer(self.asr_model, 16000)
            rec.SetSpkModel(self.spk_model)

            # 流式处理音频数据
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data)

            # 获取最终声纹特征
            final_result = json.loads(rec.FinalResult())
            return np.array(final_result.get("spk", []))

    def identify_speaker(self, wav_path):
        """
        核心识别方法
        参数：
            wav_path : str - 需要识别的音频文件路径
        返回：
            tuple - (最匹配的说话人ID, 最高相似度分数)
                    当最高分低于阈值时返回 ("unknown", 最高分)
        """
        # 提取目标声纹
        target_vec = self.extract_voiceprint(wav_path)
        if len(target_vec) != 128:
            raise ValueError("声纹特征提取失败")

        # 相似度计算
        best_match = ("unknown", 0.0)
        for speaker_id, profile in self.speaker_db.items():
            # 余弦相似度计算
            similarity = np.dot(target_vec, profile) / (
                    np.linalg.norm(target_vec) * np.linalg.norm(profile)
            )

            # 更新最佳匹配
            if similarity > best_match[1]:
                best_match = (speaker_id, similarity)

        # 阈值判断
        if best_match[1] < self.threshold:
            return "unknown", best_match[1]
        return best_match

    def enroll_speaker(self, wav_path, speaker_id):
        """
        注册新说话人
        参数：
            wav_path : str - 注册音频路径
            speaker_id : str - 说话人ID
        """
        vec = self.extract_voiceprint(wav_path)
        np.save(f"voice/{speaker_id}.npy", vec)
        self.speaker_db[speaker_id] = vec


# 使用示例
if __name__ == "__main__":
    vr = VoiceprintRecognizer()

    # 识别示例
    #speaker, score = vr.identify_speaker("test.wav")
    #print(f"识别结果：{speaker} (置信度：{score:.2f})")

    # 注册新用户示例
    #vr.enroll_speaker("voice/fanwu.wav", "吴凡")