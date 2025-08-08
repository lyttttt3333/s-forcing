import json
import os

video_folder = "video/dense_168_new"
output_json = "VBench/vbench2_beta_long/VBench_full_info.json"
output_json_copy = "VBench/vbench/VBench_full_info.json"

if os.path.exists(output_json):
    with open(output_json, "r", encoding="utf-8") as f:
        prompts = json.load(f)
else:
    prompts = []


existing_prompts = set(p["prompt_en"] for p in prompts)


for filename in os.listdir(video_folder):
    if filename.lower().endswith(".mp4"):
        base = os.path.splitext(filename)[0]      
        prompt = base.rsplit('-', 1)[0] 
        prompts.append({
            "prompt_en": prompt,
            "dimension": ["overall_consistency"]
        })

# 写入更新后的 JSON 文件
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(prompts, f, indent=4)
with open(output_json_copy, "w", encoding="utf-8") as f:
    json.dump(prompts, f, indent=4)

print(f"✅ 共保存 {len(prompts)} 条记录，已写入 {output_json}")
print(f"✅ 共保存 {len(prompts)} 条记录，已写入 {output_json_copy}")