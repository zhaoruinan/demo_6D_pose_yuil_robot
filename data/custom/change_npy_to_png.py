import os

# 폴더 경로 설정
folder_path = "mask"

# 폴더 내 png 파일을 하나씩 읽어들여서 파일 이름 변경
for file_name in os.listdir(folder_path):
    if file_name.endswith("_depth.png"):
        # 파일 이름 변경
        new_file_name = file_name.replace("_depth.png", ".png")
        os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_file_name))

