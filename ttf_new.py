import torchaudio
import argparse
from tts import StepAudioTTS
from tokenizer import StepAudioTokenizer
from utils import load_audio
import os


def main():
    parser = argparse.ArgumentParser(description="StepAudio Offline Inference")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Base path for model files"
    )
    parser.add_argument(
        "--synthesis-type", type=str, default="tts", help="Use tts or Clone for Synthesis"
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Output path for synthesis audios"
    )
    args = parser.parse_args()
    os.makedirs(f"{args.output_path}", exist_ok=True)

    encoder = StepAudioTokenizer(f"{args.model_path}/Step-Audio-Tokenizer")
    tts_engine = StepAudioTTS(f"{args.model_path}/Step-Audio-TTS-3B", encoder)

    if args.synthesis_type == "tts":
        text = "（RAP）我踏上自由的征途，追逐那遥远的梦想，挣脱束缚的枷锁，让心灵随风飘荡，每一步都充满力量，每一刻都无比闪亮，自由的信念在燃烧，照亮我前进的方向!"
        output_audio, sr = tts_engine(text, "Tingting")
        torchaudio.save(f"{args.output_path}/output_tts.wav", output_audio, sr)
    else:
        clone_speaker = {"speaker":"test","prompt_text":"叫做秋风起蟹脚痒，啊，什么意思呢？就是说这秋风一起啊，螃蟹就该上市了。", "wav_path":"examples/prompt_wav_yuqian.wav"}
        text_clone = "每至春节前夕，支付宝集五福活动便如一场盛大的全民狂欢，迅速席卷全国。为赋予用户一场沉浸式的奇妙之旅，快速进入五福的世界，设计师以高效的视觉表达为笔、灵动的动画效果为墨，精心勾勒出每一处细节，只为让用户在踏入的瞬间，便能沉浸于这场愉悦且美好的视觉盛宴里。"
        output_audio, sr = tts_engine(text_clone, "",clone_speaker)
        torchaudio.save(f"{args.output_path}/output_clone.wav", output_audio, sr)

if __name__ == "__main__":
    main()
