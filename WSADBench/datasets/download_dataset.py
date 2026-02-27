import argparse
import json
import os
import subprocess
import glob
import urllib

from modelscope.hub.snapshot_download import snapshot_download
import re

PART_COUNT = {
    'CV_by_I3D/UCF_Crime': 5,
    'CV_by_I3D/XD-violence': 8,
    'CV_by_MViT_32/UCF_Crime': 2,
    'CV_by_MViT_32/XD-violence': 3,
    'CV_by_SlowFast/UCF_Crime': 5,
    'CV_by_SlowFast/XD-violence': 9,
    'CV_by_SlowFast_R50/UCF_Crime': 5,
    'CV_by_SlowFast_R50/XD-violence': 9,
    'CV_by_X3DM/UCF_Crime': 3,
    'CV_by_X3DM/XD-violence': 5,
}

# ADBench
JIHULAB_DATASETS = [
    'CV_by_ResNet18',
    'CV_by_ViT',
    'Classical',
    'NLP_by_BERT',
    'NLP_by_RoBERTa'
]
def process_jihulab_dataset(dataset_prefix, target_dir):
    """
    Process JihuLab dataset: Read original author's JSON -> Download .npz files via HTTP one by one
    """
    print(f"\n" + "=" * 60)
    print(f"Start processing JihuLab dataset: {dataset_prefix}")
    print("=" * 60)

    final_path = os.path.join(target_dir, dataset_prefix)
    os.makedirs(final_path, exist_ok=True)

    url_repo = 'https://jihulab.com/BraudoCC/ADBench_datasets/-/raw/339d2ab2d53416854f6535442a67393634d1a778'
    json_url = f"{url_repo}/datasets_files_name.json"
    json_path = os.path.join(target_dir, 'datasets_files_name.json')

    # 1. Get JSON directory dictionary containing all file names
    if not os.path.exists(json_path):
        print("Fetching JihuLab dataset file list (datasets_files_name.json)...")
        try:
            urllib.request.urlretrieve(json_url, json_path)
        except Exception as e:
            print(f"Failed to download file list JSON: {e}")
            return

    # 2. Read and parse JSON
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            loaded_dict = json.load(f)
    except Exception as e:
        print(f" Failed to read JSON: {e}")
        return

    # 3. Verify if the requested folder exists in JSON
    if dataset_prefix not in loaded_dict:
        print(f"Error: The requested folder '{dataset_prefix}' does not exist in JihuLab records.")
        return

    datasets_list = loaded_dict[dataset_prefix]
    print(f"Found {len(datasets_list)} files under directory {dataset_prefix}, ready to sync...")

    # 4. Download files one by one, automatically skip existing ones
    for file_name in datasets_list:
        save_path = os.path.join(final_path, file_name)
        if os.path.exists(save_path):
            print(f"{file_name} already exists, skipping download.")
            continue

        url = f"{url_repo}/{dataset_prefix}/{file_name}"
        print(f"Downloading: {file_name} ...")
        try:
            urllib.request.urlretrieve(url, save_path)
            # print(f"✅ {file_name} downloaded successfully!") # Comment out this line to avoid log spamming, keep only downloading prompts
        except Exception as e:
            print(f"Failed to download {file_name}: {e}")

    print(f"All files under directory {dataset_prefix} have been synced successfully!")

def run_cmd(cmd):
    subprocess.run(cmd, shell=True, check=True)


def process_modelscope_dataset(dataset_prefix, target_dir):
    """
    Process a single dataset: Download -> Verify -> Merge -> Decompress -> Delete
    Example of dataset_prefix: 'CV_by_MViT_32/XD-violence'
    """
    print(f"\n" + "=" * 60)
    print(f"Start processing dataset: {dataset_prefix}")
    print("=" * 60)
    # Get relative path directory and base name
    sub_dir = os.path.join(target_dir, os.path.dirname(dataset_prefix))
    base_name = os.path.basename(dataset_prefix)
    # =========================================================
    # 0. Pre-check phase: Skip directly if the decompressed folder already exists!
    # =========================================================
    extracted_folder_path = os.path.join(sub_dir, base_name)
    if os.path.exists(extracted_folder_path):
        print(f"Detected final data directory already exists: {extracted_folder_path}")
        print(f"This dataset was downloaded and decompressed previously, no need for reprocessing, skip directly!")
        return

    # 1. Download phase (ModelScope automatically skips existing and complete files)
    pattern = f"{dataset_prefix}.*"
    try:
        dataset_dir = snapshot_download(
            repo_id='mac4mac/WSADBench-Datasets',
            repo_type='dataset',
            local_dir=target_dir,
            allow_patterns=[pattern]
        )
    except Exception as e:
        print(f"Download failed: {e}")
        return

    # 2. Locate downloaded files
    sub_dir = os.path.join(target_dir, os.path.dirname(dataset_prefix))
    base_name = os.path.basename(dataset_prefix)

    search_pattern = os.path.join(sub_dir, f"{base_name}.*")
    downloaded_files = glob.glob(search_pattern)

    if not downloaded_files:
        print(f"Warning: No files matching {pattern} were found, the file may not exist in the cloud.")
        return

    # =========================================================
    # Core fix: Use regex to match split files ending with numbers (e.g., .tar00, .tar.gz01)
    # =========================================================
    part_files = sorted([f for f in downloaded_files if re.search(r'\.(tar|tar\.gz)\d+$', f)])

    archive_path = ""

    # 3. Verification and merging phase
    if part_files:
        expected_count = PART_COUNT.get(dataset_prefix)
        if expected_count is None:
            print(f"Error: Split files detected, but the expected count for {dataset_prefix} is not configured in the PART_COUNT dictionary!")
            return

        actual_count = len(part_files)
        print(f"Verifying split files: Expected {expected_count}, found {actual_count}.")

        if actual_count != expected_count:
            print(f"Critical error: {dataset_prefix} split files are incomplete! Please re-run the script to attempt resumable download.")
            return

        archive_path = re.sub(r'\d+$', '', part_files[0])

        if not os.path.exists(archive_path):
            print(f"Merging {actual_count} split files into: {os.path.basename(archive_path)} ...")
            parts_str = " ".join(part_files)
            run_cmd(f"cat {parts_str} > {archive_path}")
        else:
            print(f"Merged file already exists, skipping merge.")

        # Delete split fragments to save space
        print("Cleaning up split fragments with numeric suffixes...")
        run_cmd(f"rm -f {' '.join(part_files)}")

    else:
        # Single compressed package
        archive_path = downloaded_files[0]

    # 4. Decompression phase
    if not archive_path.endswith(('.tar', '.tar.gz')):
        print(f"{archive_path} is not a compressed package, skipping decompression.")
        return

    print(f"Decompressing: {os.path.basename(archive_path)} ...")
    try:
        # Extract to the corresponding subdirectory
        run_cmd(f"tar -xf {archive_path} -C {sub_dir}")
        print(f"Decompression successful!")

        # 5. Delete original compressed package
        os.remove(archive_path)
        print(f"Deleted original compressed package: {os.path.basename(archive_path)} to free up space.")

    except Exception as e:
        print(f"Decompression failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-click streaming download and decompression of WSADBench dataset")
    parser.add_argument(
        "--datasets",
        nargs='+',
        required=True,
        help="Specify the list of dataset prefixes to download. For example: CV_by_MViT_32/shanghaitech CV_by_MViT_32/XD-violence"
    )
    args = parser.parse_args()

    # Get the directory where the script is located (force lock to current directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not script_dir.endswith('WSADBench/datasets'):
        print("Warning: It is recommended to place this script in the WSADBench/datasets/ directory to run.")

    target_dir = script_dir
    print(f"Working directory set to: {target_dir}")

    expanded_datasets = []
    for user_input in args.datasets:
        # Remove possible trailing slashes from input to prevent matching failure
        clean_input = user_input.rstrip('/')
        if clean_input in ['CV_by_MViT_32', 'CV_by_SlowFast', 'CV_by_SlowFast_R50', 'CV_by_X3DM', 'CV_by_I3D']:  # Classical_bags_inexact.tar.gz

            expanded_datasets.extend([f"{clean_input}/{ds_name}" for ds_name in ['UCF_Crime', 'XD-violence',  'TAD', 'shanghaitech']])
            print(f"Folder identifier recognized: {clean_input}, automatically expanded to 4 datasets.")

        elif clean_input == 'ADBench':       # pull JihuLab
            expanded_datasets.extend(JIHULAB_DATASETS)
            print(f"Recognized super macro 'ADBench', automatically expanded to {len(JIHULAB_DATASETS)} Jihu feature datasets.")
        elif clean_input == 'WSAD':  # pull WSAD
            for ds_name in ['UCF_Crime', 'XD-violence',  'TAD', 'shanghaitech']:
                for model in ['CV_by_MViT_32', 'CV_by_SlowFast', 'CV_by_SlowFast_R50', 'CV_by_X3DM', 'CV_by_I3D']:
                    expanded_datasets.append(f"{model}/{ds_name}")
            expanded_datasets.extend(['Classical_bags_inexact', 'CV_by_ResNet18_OOD'])
            print(f"Recognized super macro 'WSAD', automatically expanded to all WSAD datasets.")

        else:
            expanded_datasets.append(clean_input)
    print(f"Expanded dataset list: {expanded_datasets}")

    for ds_prefix in expanded_datasets:
        if ds_prefix in JIHULAB_DATASETS:
            process_jihulab_dataset(ds_prefix, target_dir)
        else:
            process_modelscope_dataset(ds_prefix, target_dir)
    print("\n All requested datasets have been processed!")