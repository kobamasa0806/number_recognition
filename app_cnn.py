import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision.transforms as transforms

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,1)
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(12*12*64,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = self.conv1(x)
        x = f.relu(x)
        x = self.conv2(x)
        x = f.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(-1,12*12*64)
        x = self.fc1(x)
        x = f.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return f.log_softmax(x, dim=1)

device = 'cpu'
net = MyNet().to(device)
net.load_state_dict(torch.load('model_cnn.pt', map_location=torch.device(device), weights_only=True))
net.eval()
def normalize_image(image_tensor):
    mean = 0.1307  # MNISTの平均
    std = 0.3081   # MNISTの標準偏差
    normalized_image = (image_tensor - mean) / std
    return normalized_image
# Streamlit app タイトル
st.title("手書き数字認識アプリ")
st.write("### 0～9までの手書きの数字を認識するアプリです。  \n ### 下の枠の中に一桁の数字を書いて下さい。  \n ### 書いたらその下のアップロードボタンを押して下さい。AIが認識してくれます。")
stroke_width = st.sidebar.slider("線の太さ: ", 20, 60, 30)
realtime_update = st.sidebar.checkbox("Update in realtime", False)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color="black",
    background_color="white",
    #background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=280,
    width=280,
    drawing_mode="freedraw",
    display_toolbar=st.sidebar.checkbox("Display toolbar", True),
    key="canvas",
)

# 描画した文字を整形し、CNNを使用して分類
if canvas_result.image_data is not None and len(canvas_result.json_data["objects"]) > 0:
    drawn_image = canvas_result.image_data
    drawn_image_gray = np.uint8((drawn_image[:, :, 0:3].sum(axis=2))/3)
    drawn_image_gray = Image.fromarray(drawn_image_gray)
    #print(drawn_image_gray)
    resized_image = drawn_image_gray.resize((28, 28))
    inverted_image = ImageOps.invert(resized_image)
    image_for_judge = np.array(inverted_image)/255
    #print(image_for_judge)
    image_for_judge = torch.tensor(image_for_judge, dtype=torch.float32)
    image_for_judge = image_for_judge.view(-1, 28, 28).to(device)
    image_for_judge = normalize_image(image_for_judge)
    with torch.no_grad():
        output = net(image_for_judge)
        probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()
        prediction_label = np.argmax(probabilities)
        
    st.write(f"## あなたの書いた数字は「*{prediction_label}*」ですか？")
    #plt.imshow(image_for_judge.detach().to('cpu').numpy().reshape(28, 28), cmap='gray')
    # スコアを棒グラフで表示
    fig, ax = plt.subplots()
    ax.bar(range(10), probabilities, tick_label=range(10))
    ax.set_title("各数字に対する確率")
    ax.set_xlabel("数字")
    ax.set_ylabel("確率")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y")
    st.pyplot(fig)
else:
    st.write(f"## 一桁の数字を記入してアップロードして下さい。")
