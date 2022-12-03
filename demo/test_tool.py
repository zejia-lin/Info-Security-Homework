import numpy as np
import cv2


def psnr(img1, img2):
    mse = np.mean((img1 - img2)**2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 255
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def rotate_about_center(src, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    return cv2.warpAffine(src,
                          rot_mat, (int(np.ceil(nw)), int(np.ceil(nh))),
                          flags=cv2.INTER_LANCZOS4)


def attack(img, atype):
    if atype == "ori":
        return img

    if atype.startswith("blur"):
        ksize = int(atype[len('blur'):])
        kernel = np.ones((ksize, ksize), np.float32) / (ksize * ksize)
        return cv2.filter2D(img, -1, kernel)

    if atype.startswith("rotate"):
        degree = int(atype[len('rotate'):])
        return rotate_about_center(img, degree)

    if atype.startswith("chop"):
        w, h = img.shape[:2]
        factor = int(atype[4:]) / 100
        wa = int(w * factor)
        ha = int(w * factor)
        return np.pad(img[int(wa):-int(wa), int(ha):-int(ha)],
                      ((ha, ha), (wa, wa), (0, 0)))

    if atype == "gray":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if atype == "saltnoise":
        for k in range(1000):
            i = np.random.randint(img.shape[1])
            j = np.random.randint(img.shape[0])
            if img.ndim == 2:
                img[j, i] = np.random.randint(255)
            elif img.ndim == 3:
                img[j, i, 0] = np.random.randint(255)
                img[j, i, 1] = np.random.randint(255)
                img[j, i, 2] = np.random.randint(255)
        return img

    if atype == "cover":
        cv2.rectangle(img, (384, 0), (510, 128), (0, 255, 0), 3)
        cv2.rectangle(img, (0, 0), (300, 128), (255, 0, 0), 3)
        cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5)
        cv2.line(img, (0, 511), (511, 0), (255, 0, 255), 5)
        cv2.circle(img, (256, 256), 63, (0, 0, 255), -1)
        cv2.putText(img, 'LZJ 22214373', (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 0), 2)
        return img

    if atype == "brighter200":
        img = img.astype(np.float32)
        img *= 2

        return img

    if atype == "darker50":
        img = img.astype(np.float32)
        img *= 0.5
        return img

    if atype == "largersize":
        w, h = img.shape[:2]
        return cv2.resize(img, (int(h * 1.5), w))

    if atype == "smallersize":
        w, h = img.shape[:2]
        return cv2.resize(img, (int(h * 0.5), w))

    return img


attack_list = {}
attack_list['ori'] = '原图'
attack_list['gray'] = '灰度'
attack_list['blur5'] = '模糊5'
attack_list['blur10'] = '模糊10'
attack_list['blur20'] = '模糊20'
attack_list['rotate180'] = '旋转180度'
attack_list['rotate90'] = '旋转90度'
attack_list['rotate45'] = '旋转45度'
attack_list['chop30'] = '剪切掉30%'
attack_list['chop50'] = '剪切掉50%'
attack_list['chop80'] = '剪切掉80%'
attack_list['saltnoise'] = '椒盐噪声'
attack_list['cover'] = '随机遮挡'
attack_list['brighter200'] = '亮度提高200%'
attack_list['darker50'] = '亮度降低50%'
attack_list['largersize'] = '图像拉伸'
attack_list['smallersize'] = '图像缩小'
