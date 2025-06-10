import os
import json
import imagesize
import fire
from imgutils.tagging import get_wd14_tags
from imgutils.tagging import tags_to_text

# Function to read existing tags from a file
def read_existing_tags(txt_file):
    if not os.path.exists(txt_file):
        return []
    with open(txt_file, 'r') as f:
        line = f.readline().strip()
    return line.split(',')

# Function to write combined tags to a file
def write_combined_tags(txt_file, combined_tags):
    with open(txt_file, 'w') as f:
        f.write(','.join(sorted(combined_tags)))

# Function to normalize tags
def normalize_tags(tags):
    normalized_tags = set()
    for tag in tags:
        if '_' in tag:
            tag = tag.replace('_', ' ')
        normalized_tags.add(tag)
    return normalized_tags

def process_images(
        path="data_path",
        trigger_token="test",
        prepend_tags=None,                 
        output="./test/output.json"):

    if prepend_tags is None:
        prepend_tags = []
    elif isinstance(prepend_tags, str):
        prepend_tags = [t.strip() for t in prepend_tags.split(',')]
    else: 
        prepend_tags = list(prepend_tags)

    result, MAX_SIZE = {}, 1024
    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.webp', '.jpeg', '.gif', '.bmp')):
                img_path = os.path.join(root, file)
                w, h = imagesize.get(img_path)
                w = max(64, min(MAX_SIZE, w // 64 * 64))
                h = max(64, min(MAX_SIZE, h // 64 * 64))

                rating, feats, chars = get_wd14_tags(img_path)
                new_tags = normalize_tags(tags_to_text(feats).split(', '))

                tags_ordered = [trigger_token] + prepend_tags + sorted(new_tags)
                result[img_path] = {
                    "train_resolution": [w, h],
                    "tags": ", ".join(tags_ordered)
                }

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {output}")

# Enable CLI usage with Fire
if __name__ == "__main__":
    fire.Fire(process_images)