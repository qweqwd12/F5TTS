import os
import torch
import soundfile as sf
import jiwer
from bert_score import score as bert_score
from omegaconf import OmegaConf
from f5_tts.model.backbones.dit import DiT
from hydra.utils import get_class
from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process, get_tokenizer, tokenize_batch_text
from importlib.resources import files
from f5_tts.diffusion_sampler import DiffusionSampler,DiTSamplerWrapper

class ZeroShotTTS:
    def __init__(self, ckpt_path, config_path, device=None):
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"

         # 加载配置文件
        model_cfg = OmegaConf.load(config_path)

        # 构建模型参数
        arch_params = model_cfg.model.arch

        model_kwargs = {
            'dim': arch_params.dim,
            'depth': arch_params.depth,
            'heads': arch_params.heads,
            'ff_mult': arch_params.ff_mult,
            'text_dim': arch_params.text_dim,
            'text_mask_padding': arch_params.text_mask_padding,
            'conv_layers': arch_params.conv_layers,
            'pe_attn_head': arch_params.pe_attn_head,
            'checkpoint_activations': arch_params.checkpoint_activations,

            # 默认值或固定参数
            'mel_dim': 100,
            'text_num_embeds': 256,
            'qk_norm': None,
            'long_skip_connection': False,
            'dim_head': 64,
            'dropout': 0.1
        }

        # 实例化模型
        self.model = DiT(**model_kwargs).to(self.device)
        # 将模型包装成 DiTSamplerWrapper
        self.model = DiTSamplerWrapper(
            self.model,
            num_steps=1000,
            beta_schedule="cosine",
            device=self.device
        )

        # 加载模型权重
        print("[+] 正在加载你自己的训练模型权重...")
        state_dict = torch.load(ckpt_path, map_location="cpu",weights_only=False)

        # 选择 EMA 或普通权重
        state_dict_key = "ema_model_state_dict"
        # 尝试加载权重，忽略不匹配的键
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict[state_dict_key], strict=False)

        print(f"[!] Missing keys: {missing_keys}")
        print(f"[!] Unexpected keys: {unexpected_keys}")

        self.model.eval()

        # 设置其他参数
        self.mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
        self.target_sample_rate = model_cfg.model.mel_spec.target_sample_rate
        ode_method = model_cfg.model.get('ode_method', 'euler')

        # 加载 vocoder（声码器）
        self.vocoder = load_vocoder(
            is_local=True,
            local_path="/root/autodl-tmp/F5-TTS/ckpts/vocos",
            device=self.device
        )

    
    def synthesize(self, text, output_path):
        """
        不需要参考语音和参考文本的情况下进行语音合成。
        
        :param text: 需要合成的文本内容。
        :param output_path: 输出音频文件的路径。
        """
        # 分词
        tokenizer,_ = get_tokenizer("pinyin")  # 根据你的配置设置分词器
        tokenized_text = tokenize_batch_text(tokenizer, [text])[0].to(self.device)

        # 获取长度估计（这里可以替换为 length regulator）
        duration = int(len(text) * 3)  # 粗略估算 token to frame 的映射

        dummy_cond = torch.randn(1, duration, 100).to(self.device)  # 可以改为更合理的条件

        with torch.inference_mode():
            print("[+] 开始使用 DDIM 采样生成mel频谱...")
            mel, _ = self.model.sample(
                cond=dummy_cond,
                text=tokenized_text,
                duration=duration,
                steps=32,  # 或者从参数传入
                cfg_strength=2.0,
                sway_sampling_coef=-1,
            )

            print("[+] 使用 vocoder 解码为音频...")
            wav = self.vocoder.decode(mel.transpose(1, 2))

            # 后处理音量
            target_rms = 0.1
            rms = torch.sqrt(torch.mean(wav**2))
            wav = wav * (target_rms / rms)

            wav = wav.squeeze().cpu().numpy()
            sf.write(output_path, wav, self.target_sample_rate)

        print(f"✅ 合成音频保存至：{output_path}")
        return output_path

    @staticmethod
    def evaluate(gen_text, target_text):
        wer = jiwer.wer(target_text.lower(), gen_text.lower())
        P, R, F1 = bert_score([gen_text], [target_text], lang='zh', verbose=False)
        sim = F1.item()
        return wer, sim


if __name__ == "__main__":
    # 设置路径
    ckpt_path = "/root/autodl-tmp/F5-TTS/ckpts/F5TTS_Base_vocos_pinyin_losl/model_last.pt"
    config_path = "/root/autodl-tmp/F5-TTS/ckpts/F5TTS_Base_vocos_pinyin_losl/2025-04-30/23-54-46/.hydra/config.yaml"
    output_dir = "/root/autodl-tmp/F5-TTS/tests"
    os.makedirs(output_dir, exist_ok=True)

    # 初始化合成器
    tts = ZeroShotTTS(ckpt_path, config_path)

    # 输入文本
    gen_text = "我喜欢用人工智能技术解决实际问题。"
    target_text = "我喜欢用人工智能技术解决实际问题。"  # 用于评估的目标文本

    # 合成语音
    output_wav_path = os.path.join(output_dir, "zero_shot_output.wav")
    tts.synthesize(gen_text, output_wav_path)

    # 评估 WER 和 SIM
    wer, sim = tts.evaluate(gen_text, target_text)
    print(f"[+] WER（字符错误率）: {wer:.4f}")
    print(f"[+] SIM（语义相似度 - BERTScore F1）: {sim:.4f}")