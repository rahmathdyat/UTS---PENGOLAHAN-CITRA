# UTS---PENGOLAHAN-CITRA
| NAMA | NIM |
| - | - |
| Rahmat Hidayat | 312210565 |
| Satria Dwi Aprianto | 312210490 |
| Syahril Haryanto | 312210668 |
| Farhan Zulfahriansyah | 312210494 |
| Robby Firmansyah | 312210643 |
| ALFAZA PUTRA ADJIE ARIEFIANSYAH | 312210512 |

Import Library
```
import streamlit as st
import cv2
from matplotlib import pyplot as plt
import numpy as np
```

Fungsi untuk Konversi Warna
```
def convert_rgb_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
```

Fungsi untuk Membuat Histogram
```
def plot_histogram(image):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.title('Histogram')
    return plt
```

Fungsi untuk Mengatur Kecerahan dan Kontras
```
def adjust_brightness_contrast(image, brightness=0, contrast=0):
    new_image = np.zeros(image.shape, image.dtype)
    alpha = contrast * 0.01
    beta = brightness

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
    
    return new_image
```

Fungsi untuk Mencari Kontur
```
def find_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_with_contours = image.copy()
    cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 3)
    return image_with_contours
```

Antarmuka Streamlit
```
st.title('Image Manipulation App')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, caption='Original Image', use_column_width=True)

    if st.button('Convert RGB to HSV'):
        hsv_image = convert_rgb_to_hsv(image)
        st.image(hsv_image, caption='HSV Image', use_column_width=True)

    if st.button('Show Histogram'):
        plt = plot_histogram(image)
        st.pyplot(plt)

    brightness = st.slider("Brightness", min_value=-100, max_value=100, value=0)
    contrast = st.slider("Contrast", min_value=-100, max_value=100, value=0)
    if st.button('Adjust Brightness and Contrast'):
        bc_image = adjust_brightness_contrast(image, brightness, contrast)
        st.image(bc_image, caption='Brightness and Contrast Adjusted Image', use_column_width=True)

    if st.button('Find Contours'):
        contours_image = find_contours(image)
        st.image(contours_image, caption='Image with Contours', use_column_width=True)
```

Hasil projek
![Screenshot 2024-05-30 112104](https://github.com/rahmathdyat/UTS---PENGOLAHAN-CITRA/assets/130628907/15056ac3-9ccc-470e-b81c-1d7579c61491)
![Screenshot 2024-05-30 112041](https://github.com/rahmathdyat/UTS---PENGOLAHAN-CITRA/assets/130628907/e7b1bb05-7b25-4911-b590-a414a35673bd)
![Screenshot 2024-05-30 112028](https://github.com/rahmathdyat/UTS---PENGOLAHAN-CITRA/assets/130628907/c4f41399-f516-4f3b-a175-57e341e33171)



