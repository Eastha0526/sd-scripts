from PIL import Image
import library.train_util as train_util
import glob
import os
import json
from collections import defaultdict
import argparse
from tqdm import tqdm

def get_npz_filename(data_dir, image_key, is_full_path, recursive):
    if not data_dir:
        assert is_full_path, "data_dir is required if is_full_path is False / data_dirは必須です"
        assert os.path.exists(image_key) and os.path.isabs(image_key), f"image_key must be full path / image_keyはフルパスである必要があります: {image_key}"
        image_base_name = image_key.rsplit(".", 1)[0]
        return image_base_name + ".npz"
    if is_full_path:
        base_name = os.path.splitext(os.path.basename(image_key))[0]
        relative_path = os.path.relpath(os.path.dirname(image_key), data_dir)
    else:
        base_name = image_key
        relative_path = ""

    if recursive and relative_path:
        return os.path.join(data_dir, relative_path, base_name) + ".npz"
    else:
        return os.path.join(data_dir, base_name) + ".npz"

def main(args):
    max_reso = tuple([int(t) for t in args.max_resolution.split(",")])
    bucket_manager = train_util.BucketManager(
        args.bucket_no_upscale, max_reso, args.min_bucket_reso, args.max_bucket_reso, args.bucket_reso_steps
    )
    metadata = defaultdict(dict)
    if args.json_pattern:
        json_list = glob.glob(args.json_pattern)
        if len(json_list) == 0:
            print(f"no json file found: {args.json_pattern}")
            return
        print(f"found {len(json_list)} json files.")
        image_paths = []
        for json_path in json_list:
            print(f"loading metadata: {json_path}")
            with open(json_path, "rt", encoding="utf-8") as f:
                partial_metadata = json.load(f)
                # key, value = image_path, tag
                image_paths.extend(
                    [key for key in partial_metadata.keys()]
                )
                metadata.update(
                    {
                        key : {"tag": value} for key, value in partial_metadata.items()
                    }
                )
    else:
        assert args.train_data_dir_path, "train_data_dir_path is required."
        image_paths = [str(p) for p in train_util.glob_images_pathlib(args.train_data_dir_path, args.recursive)]
    print(f"Validating paths for {len(image_paths)} paths")
    image_path_cleaned = []
    for path in tqdm(image_paths):
        if os.path.exists(path):
            image_path_cleaned.append(path)
    image_paths = image_path_cleaned
    print(f"found {len(image_paths)} images.")
    # cleanup metadata for not found images
    del image_path_cleaned
    if len(metadata) != len(image_paths):
        print("Cleaning up metadata")
        for image_key in tqdm(metadata.keys()):
            if image_key not in image_paths:
                print(f"image not found: {image_key}")
                del metadata[image_key]
    img_ar_errors = []
    bucket_counts = {}
    bucket_manager.make_buckets()
    for image_key in tqdm(image_paths, total=len(image_paths)):
        image = Image.open(image_key)
        image_path = image_key
        reso, resized_size, ar_error = bucket_manager.select_bucket(image.width, image.height)
        metadata[image_key]["train_resolution"] = (reso[0] - reso[0] % 8, reso[1] - reso[1] % 8)
        img_ar_errors.append(abs(ar_error))
    with open(args.output_json, "wt", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir_path", type=str, default=None)
    parser.add_argument("--json_pattern", type=str, default=None)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--bucket_no_upscale", type=bool, default=False)
    parser.add_argument("--max_resolution", type=str, default="1024,1024")
    parser.add_argument("--min_bucket_reso", type=int, default=256)
    parser.add_argument("--max_bucket_reso", type=int, default=4096)
    parser.add_argument("--bucket_reso_steps", type=int, default=64)
    parser.add_argument("--recursive", type=bool, default=True)
    args = parser.parse_args()
    main(args)
