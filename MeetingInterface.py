import gradio as gr
import threading
import time
import random
from dataclasses import dataclass


# ====== 数据结构 ======
@dataclass
class Message:
    speaker: str
    text: str
    timestamp: float = time.time()


# ====== 全局配置 ======
SPEAKERS = {
    "主持人": {
        "avatar": "https://img2.baidu.com/it/u=1424382294,643570206&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=506",
        # 金色麦标头像
        "color": "#ffd700",
        "position": "left"
    },
    "技术专家": {
        "avatar": "https://img0.baidu.com/it/u=552024851,2385678488&fm=253&fmt=auto&app=138&f=PNG?w=500&h=500",  # 眼镜图标
        "color": "#40e0d0",
        "position": "left"
    },
    "产品经理": {
        "avatar": "https://img2.baidu.com/it/u=1360716152,3525735271&fm=253&fmt=auto&app=138&f=JPEG?w=360&h=360",
        # 图表图标
        "color": "#9370db",
        "position": "right"
    },
    "客户代表": {
        "avatar": "https://img0.baidu.com/it/u=2699727399,3838550670&fm=253&fmt=auto&app=138&f=JPEG?w=360&h=360",
        # 用户图标
        "color": "#ff6347",
        "position": "right"
    }
}

chat_history = []
lock = threading.Lock()


# ====== 模拟会议发言线程 ======
def meeting_simulation():
    speaker_list = list(SPEAKERS.keys())
    sentence_pool = {
        "主持人": ["请技术团队说明方案", "客户方有什么建议", "我们进入下一个议题"],
        "技术专家": ["系统架构采用微服务", "需要3周完成开发", "存在技术风险点"],
        "产品经理": ["市场需求调研已完成", "建议优先开发核心功能", "需要增加预算"],
        "客户代表": ["希望增加报表功能", "交付时间需要提前", "对当前方案基本满意"]
    }

    while True:
        # 随机选择发言人
        speaker = random.choice(speaker_list)
        text = random.choice(sentence_pool[speaker])

        # 添加时间戳消息
        with lock:
            chat_history.append(
                Message(speaker=speaker, text=text, timestamp=time.time())
            )
            print(f"[{speaker}] {text}")  # 控制台日志

        time.sleep(random.uniform(1, 3))  # 随机发言间隔


# ====== 生成聊天消息HTML ======
def generate_html():
    html = '<div class="chat-container">'
    with lock:
        for msg in chat_history:
            config = SPEAKERS[msg.speaker]
            align_class = "left-align" if config["position"] == "left" else "right-align"

            html += f'''
            <div class="message-row {align_class}">
                <img src="{config['avatar']}" class="avatar" 
                     style="border: 2px solid {config['color']}">
                <div class="message-content">
                    <div class="speaker-name" style="color: {config['color']}">
                        {msg.speaker}
                    </div>
                    <div class="message-bubble" style="background: {config['color']}10; 
         border-left: 3px solid {config['color']};
         color: #333">
        {msg.text}
        <div class="timestamp">{time.strftime("%H:%M:%S", time.localtime(msg.timestamp))}</div>
    </div>
                </div>
            </div>
            '''
    # Add scroll trigger element
    html += '''
    <img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" 
         onload="this.parentNode.scrollTop = this.parentNode.scrollHeight" 
         style="display: none;">
    '''
    return html + '</div>'


# ====== 界面构建 ======
css = """
.chat-container {
    height: 70vh;
    overflow-y: auto;
    padding: 20px;
    background: #f8f9fa;
    border-radius: 10px;
}

.message-row {
    display: flex;
    margin: 15px 0;
    gap: 15px;
}

.left-align {
    flex-direction: row;
}

.right-align {
    flex-direction: row-reverse;
}

.avatar {
    width: 45px;
    height: 45px;
    border-radius: 50%;
    object-fit: cover;
}

.message-content {
    max-width: 70%;
}

.speaker-name {
    font-weight: 600;
    margin-bottom: 5px;
    font-size: 0.9em;
}

.message-bubble {
    padding: 12px 16px;
    border-radius: 12px;
    background: white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.timestamp {
    font-size: 0.8em;
    color: #666;
    margin-top: 8px;
    text-align: right;
}
"""

with gr.Blocks(css=css, title="多人会议系统") as demo:
    # 标题区
    gr.Markdown("## 多角色会议系统 ｜ 当前参会人：")
    with gr.Row():
        for speaker in SPEAKERS:
            gr.Image(SPEAKERS[speaker]["avatar"],
                     label=speaker,
                     width=30,
                     show_label=True)

    # 聊天区
    chat_display = gr.HTML()


    # 自动更新
    def update_chat():
        while True:
            time.sleep(0.3)
            yield generate_html()


    demo.load(
        update_chat,
        outputs=chat_display,
        every=0.3
    )

# ====== 启动系统 ======
if __name__ == "__main__":
    threading.Thread(target=meeting_simulation, daemon=True).start()
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True
    )
