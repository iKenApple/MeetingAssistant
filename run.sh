python3 MeetingAssistant.py --tokens=models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt --encoder=models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx  --decoder=models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx --joiner=models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx --speaker-file ./speaker.txt --speaker-model models/wespeaker_zh_cnceleb_resnet34.onnx --silero-vad-model=models/silero_vad.onnx --provider=coreml