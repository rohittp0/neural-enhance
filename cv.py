import numpy as np
import cv2


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened


def denoise(img):
    show_img = cv2.fastNlMeansDenoisingColored(img, templateWindowSize=5, searchWindowSize=21, h=8, hColor=10)
    show_img = unsharp_mask(show_img, kernel_size=(5, 5), sigma=1.5, amount=1.2, threshold=0)

    return show_img

# cv2.imwrite('24.png', denoise(cv2.imread('./data/1-2383/y/24.png')))

# python enhance.py --train "data/output/*/x/*.png" --type photo --model repair --epochs=500 --batch-shape=64 --device=gpu0  --generator-downscale=2 --generator-upscale=2 --generator-blocks=8 --generator-filters=128 --generator-residual=0  --perceptual-layer=conv2_2 --smoothness-weight=1e3 --adversary-weight=0.0
# python enhance.py --type=photo --model=repair --zoom=1 --out 24.png ./data/1-2383/y/24.png
