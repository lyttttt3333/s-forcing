import os
import zipfile

def is_video_file(filename):
    video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    return os.path.splitext(filename)[1].lower() in video_exts

def collect_top_k_videos(main_folder, k):
    selected_files = []

    for subfolder in sorted(os.listdir(main_folder)):
        subfolder_path = os.path.join(main_folder, subfolder)
        if os.path.isdir(subfolder_path):
            video_files = [f for f in os.listdir(subfolder_path) if is_video_file(f)]
            video_files.sort()
            top_k = video_files[:k]
            for video in top_k:
                full_path = os.path.join(subfolder_path, video)
                filename, ext = os.path.splitext(video)
                # 生成新的唯一名称
                unique_name = f"{filename}__{subfolder}{ext}"
                selected_files.append((full_path, unique_name))

    return selected_files

def zip_files(file_tuples, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w') as zipf:
        for full_path, unique_name in file_tuples:
            zipf.write(full_path, arcname=unique_name)

if __name__ == "__main__":
    main_folder = "/home/jxiao15/Genex_mem/videoacc/Self-Forcing/video"   
    k = 12                                 
    output_zip = f"forcing_video_top_{k}.zip"

    files_to_zip = collect_top_k_videos(main_folder, k)
    print(files_to_zip)
    zip_files(files_to_zip, output_zip)
    print(f"打包完成，共打包 {len(files_to_zip)} 个文件 -> {output_zip}")