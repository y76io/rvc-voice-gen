import subprocess
import os

def run_inference(
    model_name: str,
    input_path: str,
    output_path: str,
    export_format: str = "WAV",
    f0_method: str = "rmvpe",
    f0_up_key: int = 0,
    filter_radius: int = 3,
    rms_mix_rate: float = 0.8,
    protect: float = 0.5,
    index_rate: float = 0.7,
    hop_length: int = 128,
    clean_strength: float = 0.7,
    split_audio: bool = False,
    clean_audio: bool = False,
    f0_autotune: bool = False,
    formant_shift: bool = False,
    formant_qfrency: float = 1.0,
    formant_timbre: float = 1.0,
    embedder_model: str = "contentvec",
    embedder_model_custom: str = "",
    # Post-processing effects
    post_process: bool = False,
    reverb: bool = False,
    pitch_shift: bool = False,
    limiter: bool = False,
    gain: bool = False,
    distortion: bool = False,
    chorus: bool = False,
    bitcrush: bool = False,
    clipping: bool = False,
    compressor: bool = False,
    delay: bool = False,
    # Effect parameters
    reverb_room_size: float = 0.5,
    reverb_damping: float = 0.5,
    reverb_wet_gain: float = 0.0,
    reverb_dry_gain: float = 0.0,
    reverb_width: float = 1.0,
    reverb_freeze_mode: float = 0.0,
    pitch_shift_semitones: float = 0.0,
    limiter_threshold: float = -1.0,
    limiter_release_time: float = 0.05,
    gain_db: float = 0.0,
    distortion_gain: float = 0.0,
    chorus_rate: float = 1.5,
    chorus_depth: float = 0.1,
    chorus_center_delay: float = 15.0,
    chorus_feedback: float = 0.25,
    chorus_mix: float = 0.5,
    bitcrush_bit_depth: int = 4,
    clipping_threshold: float = 0.5,
    compressor_threshold: float = -20.0,
    compressor_ratio: float = 4.0,
    compressor_attack: float = 0.001,
    compressor_release: float = 0.1,
    delay_seconds: float = 0.1,
    delay_feedback: float = 0.5,
    delay_mix: float = 0.5,
):
    current_dir = os.getcwd()
    model_folder = os.path.join(current_dir, f"logs/{model_name}")
    if not os.path.exists(model_folder):
        raise FileNotFoundError(f"Model directory not found: {model_folder}")

    files_in_folder = os.listdir(model_folder)
    best_pth = next((f for f in files_in_folder if f.endswith(".pth") and model_name in f and "_best_model" in f), None)
    if best_pth is not None:
        pth_path = best_pth
    else:
        pth_path = next((f for f in files_in_folder if f.endswith(".pth")and model_name in f), None)
    index_file = next((f for f in files_in_folder if f.endswith(".index")), None)

    if pth_path is None or index_file is None:
        raise FileNotFoundError("No model found.")

    pth_file = os.path.join(model_folder, pth_path)
    index_file = os.path.join(model_folder, index_file)

    cmd = [
        "python", "core.py", "infer",
        "--pitch", str(f0_up_key),
        "--filter_radius", str(filter_radius),
        "--volume_envelope", str(rms_mix_rate),
        "--index_rate", str(index_rate),
        "--hop_length", str(hop_length),
        "--protect", str(protect),
        "--f0_autotune", str(f0_autotune),
        "--f0_method", f0_method,
        "--input_path", input_path,
        "--output_path", output_path,
        "--pth_path", pth_file,
        "--index_path", index_file,
        "--split_audio", str(split_audio),
        "--clean_audio", str(clean_audio),
        "--clean_strength", str(clean_strength),
        "--export_format", export_format,
        "--embedder_model", embedder_model,
        "--embedder_model_custom", embedder_model_custom,
        "--formant_shifting", str(formant_shift),
        "--formant_qfrency", str(formant_qfrency),
        "--formant_timbre", str(formant_timbre),
        "--post_process", str(post_process),
        "--reverb", str(reverb),
        "--pitch_shift", str(pitch_shift),
        "--limiter", str(limiter),
        "--gain", str(gain),
        "--distortion", str(distortion),
        "--chorus", str(chorus),
        "--bitcrush", str(bitcrush),
        "--clipping", str(clipping),
        "--compressor", str(compressor),
        "--delay", str(delay),
        "--reverb_room_size", str(reverb_room_size),
        "--reverb_damping", str(reverb_damping),
        "--reverb_wet_gain", str(reverb_wet_gain),
        "--reverb_dry_gain", str(reverb_dry_gain),
        "--reverb_width", str(reverb_width),
        "--reverb_freeze_mode", str(reverb_freeze_mode),
        "--pitch_shift_semitones", str(pitch_shift_semitones),
        "--limiter_threshold", str(limiter_threshold),
        "--limiter_release_time", str(limiter_release_time),
        "--gain_db", str(gain_db),
        "--distortion_gain", str(distortion_gain),
        "--chorus_rate", str(chorus_rate),
        "--chorus_depth", str(chorus_depth),
        "--chorus_center_delay", str(chorus_center_delay),
        "--chorus_feedback", str(chorus_feedback),
        "--chorus_mix", str(chorus_mix),
        "--bitcrush_bit_depth", str(bitcrush_bit_depth),
        "--clipping_threshold", str(clipping_threshold),
        "--compressor_threshold", str(compressor_threshold),
        "--compressor_ratio", str(compressor_ratio),
        "--compressor_attack", str(compressor_attack),
        "--compressor_release", str(compressor_release),
        "--delay_seconds", str(delay_seconds),
        "--delay_feedback", str(delay_feedback),
        "--delay_mix", str(delay_mix),
    ]

    subprocess.run(cmd, check=True)
    print(f"Inference complete. Output file at: {output_path}")
