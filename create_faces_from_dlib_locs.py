import os
import json
import cv2
from tqdm import tqdm

compressions = ["c40", "c23",  "masks"] # "masks" "raw", "c23", "c40", , "masks"
datasets = ["Deepfakes" ,"FaceSwap" , "NeuralTextures", "Face2Face"] #
dataset = datasets[1]
# dataset = "real"
print(dataset)
dataset_path = "/media/user/deepfake/data/FaceForensics/manipulated_sequences"
# dataset_path = "/media/user/deepfake/data/FaceForensics/original_sequences/youtube"
# save_path = "/media/user/deepfake/data/new_face"
save_path = "/home/user/ff/face"

json_file = "./FFD/Pytorch_Retinaface/dlib_{}.json".format(dataset)
with open(json_file,'r') as load_f:
    videos_face_locs = json.load(load_f)

# video_path = "/media/user/deepfake/data/FaceForensics/original_sequences/youtube/raw/videos/000.mp4"

def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_video(video_path, output_folder):
    make_dirs(output_folder)
    reader = cv2.VideoCapture(video_path)
    video_name = video_path.split('/')[-1].split('.')[0]
    locs = videos_face_locs[video_name]
    max_frame = max([int(i) for i in locs.keys()])
    frame_num = 0
    # print(sorted(locs.keys()))

    while reader.isOpened():
        # print(image_name)
        success, image = reader.read()
        if not success:
            break
        if frame_num > max_frame:
            break
        if str(frame_num) in locs.keys():
            x, y, size = locs[str(frame_num)]
            cropped_face = image[y:y+size, x:x+size]
            cv2.imwrite(os.path.join(output_folder, str(frame_num)+".png"), cropped_face)
        frame_num += 1
    reader.release()

def extract_videos(video_folder, output_folder):
    for video_fn in tqdm(os.listdir(video_folder)):
        video_name = video_fn.split('.')[0]
        video_path = os.path.join(video_folder, video_fn)
        save_folder = os.path.join(output_folder, video_name)
        extract_video(video_path, save_folder)


if __name__ == "__main__":
    for compression in compressions:
        video_folder = os.path.join(dataset_path, dataset, compression, "videos")
        # video_folder = os.path.join(dataset_path, compression, "videos")
        output_folder = os.path.join(save_path, dataset, compression)
        extract_videos(video_folder, output_folder)
    #  extract_videos("/home/user/ff/real", "/home/user/ff/images")
    