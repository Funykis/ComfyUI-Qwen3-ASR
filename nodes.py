import os
import shutil
import torch
import numpy as np
import folder_paths
import comfy.model_management as mm
from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import split_audio_into_chunks

# Register Qwen3-ASR models folder with ComfyUI
QWEN3_ASR_MODELS_DIR = os.path.join(folder_paths.models_dir, "Qwen3-ASR")
os.makedirs(QWEN3_ASR_MODELS_DIR, exist_ok=True)
folder_paths.add_model_folder_path("Qwen3-ASR", QWEN3_ASR_MODELS_DIR)

# Model repo mappings
QWEN3_ASR_MODELS = {
    "Qwen/Qwen3-ASR-1.7B": "Qwen3-ASR-1.7B",
    "Qwen/Qwen3-ASR-0.6B": "Qwen3-ASR-0.6B",
}

QWEN3_FORCED_ALIGNERS = {
    "None": None,
    "Qwen/Qwen3-ForcedAligner-0.6B": "Qwen3-ForcedAligner-0.6B",
}

# Supported languages
SUPPORTED_LANGUAGES = [
    "auto",
    "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish",
    "Portuguese", "Indonesian", "Italian", "Korean", "Russian", "Thai",
    "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay", "Dutch", "Swedish",
    "Danish", "Finnish", "Polish", "Czech", "Filipino", "Persian", "Greek",
    "Hungarian", "Macedonian", "Romanian"
]


def get_local_model_path(repo_id: str) -> str:
    folder_name = QWEN3_ASR_MODELS.get(repo_id) or QWEN3_FORCED_ALIGNERS.get(repo_id) or repo_id.replace("/", "_")
    return os.path.join(QWEN3_ASR_MODELS_DIR, folder_name)


def migrate_cached_model(repo_id: str, target_path: str) -> bool:
    if os.path.exists(target_path) and os.listdir(target_path):
        return True
    
    hf_cache = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    hf_model_dir = os.path.join(hf_cache, f"models--{repo_id.replace('/', '--')}")
    if os.path.exists(hf_model_dir):
        snapshots_dir = os.path.join(hf_model_dir, "snapshots")
        if os.path.exists(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                source = os.path.join(snapshots_dir, snapshots[0])
                print(f"Migrating model from HuggingFace cache: {source} -> {target_path}")
                shutil.copytree(source, target_path, dirs_exist_ok=True)
                return True
    
    ms_cache = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub")
    ms_model_dir = os.path.join(ms_cache, repo_id.replace("/", os.sep))
    if os.path.exists(ms_model_dir):
        print(f"Migrating model from ModelScope cache: {ms_model_dir} -> {target_path}")
        shutil.copytree(ms_model_dir, target_path, dirs_exist_ok=True)
        return True
    
    return False


def download_model_to_comfyui(repo_id: str, source: str) -> str:
    target_path = get_local_model_path(repo_id)
    
    if migrate_cached_model(repo_id, target_path):
        print(f"Model available at: {target_path}")
        return target_path
    
    os.makedirs(target_path, exist_ok=True)
    
    if source == "ModelScope":
        from modelscope import snapshot_download
        print(f"Downloading {repo_id} from ModelScope to {target_path}...")
        snapshot_download(repo_id, local_dir=target_path)
    else:
        from huggingface_hub import snapshot_download
        print(f"Downloading {repo_id} from HuggingFace to {target_path}...")
        snapshot_download(repo_id, local_dir=target_path)
    
    return target_path


def load_audio_input(audio_input):
    if audio_input is None:
        return None
        
    waveform = audio_input["waveform"]
    sr = audio_input["sample_rate"]
    
    wav = waveform[0]
    
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0)
    else:
        wav = wav.squeeze(0)
        
    return (wav.numpy().astype(np.float32), sr)


def split_waveform_by_duration(wav: np.ndarray, sr: int, chunk_seconds: float):
    """
    将一段 mono waveform 按目标时长切成若干段,避免超长音频一次性喂给模型导致显存溢出。

    直接复用 qwen_asr 库内部的 split_audio_into_chunks:在每个目标切点 ±5s 窗口内,
    用 100ms 滑窗找能量最低(最接近静音/停顿)的位置下刀;切片无重叠、无缝拼接,
    过短的尾段会在末尾补零到至少 0.5s。返回值中的偏移按真实边界累计,补零不计入。

    Args:
        wav: mono float32 波形。
        sr: 采样率。
        chunk_seconds: 每段目标时长(秒)。<=0 时不切片,返回整段音频。

    Returns:
        List[Tuple[np.ndarray, float]]: [(分片波形, 该分片在原始音频中的起始秒数), ...]
    """
    chunk_seconds = float(chunk_seconds)

    if chunk_seconds <= 0:
        return [(wav, 0.0)]

    return split_audio_into_chunks(wav=wav, sr=sr, max_chunk_sec=chunk_seconds)


class Qwen3ASRLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "repo_id": (list(QWEN3_ASR_MODELS.keys()), {"default": "Qwen/Qwen3-ASR-1.7B"}),
                "source": (["HuggingFace", "ModelScope"], {"default": "HuggingFace"}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "bf16"}),
                "attention": (["auto", "flash_attention_2", "sdpa", "eager"], {"default": "auto"}),
                "max_new_tokens": ("INT", {
                    "default": 1024, "min": 64, "max": 8192, "step": 64,
                    "tooltip": "单次生成允许输出的最大 token 数(每个音频分片各自独立计算)。"
                               "该值在加载模型时固定,值过小会导致长音频/信息密度高的语言(如中文)转录文本被截断。"
                               "建议根据 Transcribe 节点里的 chunk_seconds 一起调整:分片越长,这个值也需要越大。"
                }),
                "max_inference_batch_size": ("INT", {
                    "default": 4, "min": 1, "max": 128, "step": 1,
                    "tooltip": "模型单次推理时允许合并处理的最大分片(批次)数量。"
                               "该值在加载模型时固定;调大可提升多分片/批量转录时的吞吐,但会占用更多显存。"
                }),
            },
            "optional": {
                "forced_aligner": (list(QWEN3_FORCED_ALIGNERS.keys()), {"default": "None"}),
                "local_model_path": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("QWEN3_ASR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "Qwen3-ASR"

    def load_model(self, repo_id, source, precision, attention, max_new_tokens=1024, max_inference_batch_size=4, forced_aligner="None", local_model_path=""):
        device = mm.get_torch_device()
        
        dtype = torch.float32
        if precision == "bf16":
            if device.type == "mps":
                dtype = torch.float16
                print("Note: Using fp16 on MPS (bf16 has limited support)")
            else:
                dtype = torch.bfloat16
        elif precision == "fp16":
            dtype = torch.float16
            
        if local_model_path and local_model_path.strip() != "":
            model_path = local_model_path.strip()
            print(f"Loading from local path: {model_path}")
        else:
            local_path = get_local_model_path(repo_id)
            if os.path.exists(local_path) and os.listdir(local_path):
                model_path = local_path
                print(f"Loading from ComfyUI models folder: {model_path}")
            else:
                model_path = download_model_to_comfyui(repo_id, source)
        
        model_kwargs = dict(
            dtype=dtype,
            device_map=str(device),
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )
        if attention != "auto":
            model_kwargs["attn_implementation"] = attention
            
        if forced_aligner and forced_aligner != "None":
            aligner_local = get_local_model_path(forced_aligner)
            if not (os.path.exists(aligner_local) and os.listdir(aligner_local)):
                aligner_local = download_model_to_comfyui(forced_aligner, source)
            model_kwargs["forced_aligner"] = aligner_local
            model_kwargs["forced_aligner_kwargs"] = dict(
                dtype=dtype,
                device_map=str(device),
            )
            if attention != "auto":
                model_kwargs["forced_aligner_kwargs"]["attn_implementation"] = attention
        
        print(f"Loading Qwen3-ASR model from {model_path}...")
        model = Qwen3ASRModel.from_pretrained(model_path, **model_kwargs)
        
        return (model,)


class Qwen3ASRTranscribe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_ASR_MODEL",),
                "audio": ("AUDIO",),
            },
            "optional": {
                "language": (SUPPORTED_LANGUAGES, {"default": "auto"}),
                "context": ("STRING", {"default": "", "multiline": True}),
                "return_timestamps": ("BOOLEAN", {"default": False}),
                "chunk_seconds": ("INT", {
                    "default": 60, "min": 0, "max": 1200, "step": 5,
                    "tooltip": "超过该时长的音频会先被切成多段再逐段转录,避免长音频一次性编码/解码导致显存溢出。"
                               "设为 0 则不做预切片(改为依赖 qwen_asr 库内部默认的切片阈值,不开时间戳时约 1200 秒,"
                               "开时间戳时约 180 秒)。显存较小时建议调小此值;同时请确保 Loader 节点里的 "
                               "max_new_tokens 足够覆盖单个分片的文本量,否则单段文本仍会被截断。"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "language", "timestamps")
    FUNCTION = "transcribe"
    CATEGORY = "Qwen3-ASR"

    def transcribe(self, model, audio, language="auto", context="", return_timestamps=False, chunk_seconds=60):
        audio_data = load_audio_input(audio)
        if audio_data is None:
            return ("", "", "")

        wav, sr = audio_data
        lang = None if language == "auto" else language
        ctx = context if context.strip() else ""

        # 按 chunk_seconds 预切片;短音频(或 chunk_seconds<=0)时只会得到一段,行为与切片前完全一致
        segments = split_waveform_by_duration(wav, sr, chunk_seconds)
        chunk_audios = [(seg_wav, sr) for seg_wav, _ in segments]

        results = model.transcribe(
            audio=chunk_audios,
            language=lang,
            context=ctx if ctx else None,
            return_time_stamps=return_timestamps,
        )

        # 无缝拼接各分片文本
        text = "".join(r.text for r in results)

        # 语言取各分片里出现次数最多的一个(通常只有一种,forced language 时也会一致)
        lang_counts = {}
        for r in results:
            if r.language:
                lang_counts[r.language] = lang_counts.get(r.language, 0) + 1
        detected_lang = max(lang_counts, key=lang_counts.get) if lang_counts else ""

        # 时间戳需要加上每个分片在原始音频中的起始偏移,才能对齐回完整音频的时间轴
        timestamps_str = ""
        if return_timestamps:
            ts_lines = []
            for (_, offset_sec), r in zip(segments, results):
                if r.time_stamps:
                    for ts in r.time_stamps:
                        ts_lines.append(
                            f"{ts.start_time + offset_sec:.2f}-{ts.end_time + offset_sec:.2f}: {ts.text}"
                        )
            timestamps_str = "\n".join(ts_lines)

        if len(segments) > 1:
            mm.soft_empty_cache()

        return (text, detected_lang, timestamps_str)


class Qwen3ASRBatchTranscribe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWEN3_ASR_MODEL",),
                "audio_list": ("AUDIO",),
            },
            "optional": {
                "language": (SUPPORTED_LANGUAGES, {"default": "auto"}),
                "return_timestamps": ("BOOLEAN", {"default": False}),
                "chunk_seconds": ("INT", {
                    "default": 60, "min": 0, "max": 1200, "step": 5,
                    "tooltip": "批量列表中任意一条音频若超过该时长,会先被切成多段再转录,避免显存溢出。"
                               "设为 0 则不做预切片。同一批次内所有音频共用这个设置。"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transcriptions",)
    FUNCTION = "batch_transcribe"
    CATEGORY = "Qwen3-ASR"

    def batch_transcribe(self, model, audio_list, language="auto", return_timestamps=False, chunk_seconds=60):
        if not isinstance(audio_list, list):
            audio_list = [audio_list]
            
        audio_inputs = []
        for audio in audio_list:
            audio_data = load_audio_input(audio)
            if audio_data:
                audio_inputs.append(audio_data)
        
        if not audio_inputs:
            return ("",)
        
        lang = None if language == "auto" else language

        # 把批次里每条音频各自按 chunk_seconds 预切片,展开成一个扁平的分片列表送去推理,
        # 同时记录每个分片属于第几条原始音频、以及它在原始音频里的起始秒数,方便之后合并回去
        chunk_audios = []
        chunk_langs = []
        chunk_owner = []
        chunk_offsets = []

        for orig_idx, (wav, sr) in enumerate(audio_inputs):
            for seg_wav, offset_sec in split_waveform_by_duration(wav, sr, chunk_seconds):
                chunk_audios.append((seg_wav, sr))
                chunk_langs.append(lang)
                chunk_owner.append(orig_idx)
                chunk_offsets.append(offset_sec)

        results = model.transcribe(
            audio=chunk_audios,
            language=chunk_langs if lang else None,
            return_time_stamps=return_timestamps,
        )

        # 按原始音频序号重新聚合各分片的结果:文本无缝拼接,时间戳加回分片偏移
        merged_text = ["" for _ in audio_inputs]
        merged_lang_counts = [dict() for _ in audio_inputs]
        merged_ts_lines = [[] for _ in audio_inputs]

        for owner, offset_sec, r in zip(chunk_owner, chunk_offsets, results):
            merged_text[owner] += r.text
            if r.language:
                merged_lang_counts[owner][r.language] = merged_lang_counts[owner].get(r.language, 0) + 1
            if return_timestamps and r.time_stamps:
                for ts in r.time_stamps:
                    merged_ts_lines[owner].append(
                        f"    {ts.start_time + offset_sec:.2f}-{ts.end_time + offset_sec:.2f}: {ts.text}"
                    )

        output_lines = []
        for i in range(len(audio_inputs)):
            lang_counts = merged_lang_counts[i]
            detected_lang = max(lang_counts, key=lang_counts.get) if lang_counts else ""
            output_lines.append(f"[{i}] ({detected_lang}): {merged_text[i]}")
            output_lines.extend(merged_ts_lines[i])

        mm.soft_empty_cache()

        return ("\n".join(output_lines),)
