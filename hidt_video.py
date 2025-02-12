import cv2
import numpy as np
import time

import sys
sys.path.append('/home/shumas/hidt')

import torch
from PIL import Image
from torchvision import transforms

from hidt.networks.enhancement.RRDBNet_arch import RRDBNet
from hidt.style_transformer import StyleTransformer
from hidt.utils.preprocessing import GridCrop, enhancement_preprocessing

config_path = '/home/shumas/hidt/configs/daytime.yaml'
gen_weights_path = '/home/shumas/hidt/trained_models/generator/daytime.pt'
inference_size = 256  # the network has been trained to do inference in 256px, any higher value might lead to artifacts
device = 'cuda:0'
image_path = '/home/shumas/hidt/images/daytime/content/1.jpg'
styles_path = '/home/shumas/hidt/styles.txt'
enhancer_weights = '/home/shumas/hidt/trained_models/enhancer/enhancer.pth'

style_transformer = StyleTransformer(config_path,
                                     gen_weights_path,
                                     inference_size=inference_size,
                                     device=device)
with open(styles_path) as f:
    styles = f.read()
styles = {style.split(',')[0]: torch.tensor([float(el) for el in style.split(',')[1][1:-1].split(' ')]) for style in styles.split('\n')[:-1]}
#image = Image.open(image_path)
crop_transform = GridCrop(4, 1, hires_size=inference_size * 4)

#style_to_transfer = styles['night']
style_to_transfer = styles['sunsetred']
style_to_transfer = style_to_transfer.view(1, 1, 3, 1).to(device)

# video
video_path = "day2_front_in_change.mp4"
cap = cv2.VideoCapture(video_path)

# 動画の保存設定
#frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 10)
#frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 10)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('style_change/hidt_output_day_to_sunsetred.mp4', fourcc, fps, (frame_width, frame_height))
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # コーデック（例: mp4v）
#out = cv2.VideoWriter('hidt_output.mp4', fourcc, 20.0, (640, 480))  # 保存先ファイル、フレームレート、解像度

while True:
    ret, frame = cap.read()

    if not ret:
        print("動画の最後に到達しました。")
        break

    # PIL 形式に変換
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    with torch.no_grad():
        content_decomposition = style_transformer.get_content(image)[0]
        decoder_input = {'content': content_decomposition['content'],
                         'intermediate_outputs': content_decomposition['intermediate_outputs'],
                         'style': style_to_transfer}
        transferred = style_transformer.trainer.gen.decode(decoder_input)['images']

    output_frame = transforms.ToPILImage()((transferred[0].cpu().clamp(-1, 1) + 1.) / 2.)
    #output_frame = transforms.ToPILImage()((transferred[0].cpu().clamp(0, 1)))

    #output_frame.show()
    # Pillow の Image オブジェクトを NumPy 配列に変換 (OpenCV 互換の形式に)
    image_cv = np.array(output_frame)
    # Pillow は RGB 形式なので、OpenCV の BGR に変換 (必要に応じて)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # frame_width と frame_height を指定してリサイズ
    image_cv_resized = cv2.resize(image_cv, (frame_width, frame_height))

    # フレームを保存
    out.write(image_cv_resized)

    cv2.imshow('Video Playback', image_cv_resized)

    time.sleep(0.05)

    if cv2.waitKey(1) & 0xFF == ord('q'):
    #if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
#save_path = "/home/shumas/video_capture/output.png"
#output_image.save(save_path)
#print(f"画像が保存されました: {save_path}")

