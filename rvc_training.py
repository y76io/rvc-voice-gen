import subprocess

def preprocess_dataset(model_name: str, dataset_path: str, sample_rate: int = 40000, cpu_cores: int = 2, 
                       cut_preprocess: bool = True, process_effects: bool = False, 
                       noise_reduction: bool = False, noise_reduction_strength: float = 0.7):
    cmd = [
        "python", "core.py", "preprocess",
        "--model_name", model_name,
        "--dataset_path", dataset_path,
        "--sample_rate", str(sample_rate),
        "--cpu_cores", str(cpu_cores),
        "--cut_preprocess", str(cut_preprocess),
        "--process_effects", str(process_effects),
        "--noise_reduction", str(noise_reduction),
        "--noise_reduction_strength", str(noise_reduction_strength),
    ]
    subprocess.run(cmd, check=True)
    print("Preprocessing complete.")

def extract_features(model_name: str, rvc_version: str = "v2", f0_method: str = "rmvpe",
                     hop_length: int = 128, sample_rate: int = 40000, cpu_cores: int = 2, gpu: int = 0,
                     embedder_model: str = "contentvec", embedder_model_custom: str = ""):
    cmd = [
        "python", "core.py", "extract",
        "--model_name", model_name,
        "--rvc_version", rvc_version,
        "--f0_method", f0_method,
        "--hop_length", str(hop_length),
        "--sample_rate", str(sample_rate),
        "--cpu_cores", str(cpu_cores),
        "--gpu", str(gpu),
        "--embedder_model", embedder_model,
        "--embedder_model_custom", embedder_model_custom,
    ]
    subprocess.run(cmd, check=True)
    print("Feature extraction complete.")

def train_model(model_name: str, rvc_version: str = "v2", total_epoch: int = 800, sample_rate: int = 40000,
                batch_size: int = 15, gpu: int = 0, pretrained: bool = True, custom_pretrained: bool = False,
                g_pretrained_path: str = "", d_pretrained_path: str = "", overtraining_detector: bool = True,
                overtraining_threshold: int = 50, cleanup: bool = False, cache_data_in_gpu: bool = False,
                save_every_epoch: int = 10, save_only_latest: bool = False, save_every_weights: bool = False):
    cmd = [
        "python", "core.py", "train",
        "--model_name", model_name,
        "--rvc_version", rvc_version,
        "--save_every_epoch", str(save_every_epoch),
        "--save_only_latest", str(save_only_latest),
        "--save_every_weights", str(save_every_weights),
        "--total_epoch", str(total_epoch),
        "--sample_rate", str(sample_rate),
        "--batch_size", str(batch_size),
        "--gpu", str(gpu),
        "--pretrained", str(pretrained),
        "--custom_pretrained", str(custom_pretrained),
        "--g_pretrained_path", g_pretrained_path,
        "--d_pretrained_path", d_pretrained_path,
        "--overtraining_detector", str(overtraining_detector),
        "--overtraining_threshold", str(overtraining_threshold),
        "--cleanup", str(cleanup),
        "--cache_data_in_gpu", str(cache_data_in_gpu),
    ]
    subprocess.run(cmd, check=True)
    print("Training complete.")

def create_index(model_name: str, rvc_version: str = "v2", index_algorithm: str = "Auto"):
    cmd = [
        "python", "core.py", "index",
        "--model_name", model_name,
        "--rvc_version", rvc_version,
        "--index_algorithm", index_algorithm,
    ]
    subprocess.run(cmd, check=True)
    print("Index creation complete.")

def download_model(model_link: str):
    cmd = [
        "python", "core.py", "download",
        "--model_link", model_link,
    ]
    subprocess.run(cmd, check=True)
    print("Model downloaded successfully.")
