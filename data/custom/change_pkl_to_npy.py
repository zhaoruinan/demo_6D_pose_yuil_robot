import os
import pickle
import numpy as np

# 폴더 경로 설정
folder_path = "pose"

# 폴더 내 pkl 파일을 하나씩 읽어들여서 데이터 추출 및 저장
for file_name in os.listdir(folder_path):
    if file_name.endswith("_RT.pkl"):
        # pkl 파일 로드
        with open(os.path.join(folder_path, file_name), "rb") as f:
            data = pickle.load(f)

        # 'RT' 값 추출
        RT = np.array(data["RT"], dtype=np.float32)

        # npy 파일로 저장
        new_file_name = "pose" + file_name.split("_")[0] + ".npy"
        np.save(os.path.join(folder_path, new_file_name), RT)

        # 기존 파일 삭제
        os.remove(os.path.join(folder_path, file_name))

