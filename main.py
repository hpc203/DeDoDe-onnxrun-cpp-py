import numpy as np
import onnxruntime as ort
import cv2

class DeDoDeRunner_end2end:
    def __init__(self, end2end_path, img_size=[256, 256], fp16=False, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]):
        self.end2end = ort.InferenceSession(end2end_path, providers=providers)
        self.mean_ = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1,1,3))
        self.std_ = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1,1,3))
        self.H, self.W = img_size
        self.fp16 = fp16

    def preprocess(self, image_a, image_b):
        images = np.stack([cv2.resize(image_a, (self.W, self.H)), cv2.resize(image_b, (self.W, self.H))])
        images = (images / 255.0 - self.mean_) / self.std_
        return images.transpose(0, 3, 1, 2).astype(np.float32)

    def detect(self, image_a, image_b):
        H_A, W_A = image_a.shape[:2]
        H_B, W_B = image_b.shape[:2]
        images = self.preprocess(cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB), cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB))
        if self.fp16:
            images = images.astype(np.float16)
        matches_A, matches_B, batch_ids = self.end2end.run(None, {"images": images})

        # Postprocessing
        matches_A = self.postprocess(matches_A, H_A, W_A)
        matches_B = self.postprocess(matches_B, H_B, W_B)
        return matches_A, matches_B

    def postprocess(self, matches, H, W):
        return (matches + 1) / 2 * [W, H]

def draw_matches(im_A, kpts_A, im_B, kpts_B):
    kpts_A = [cv2.KeyPoint(x, y, 1.0) for x, y in kpts_A]
    kpts_B = [cv2.KeyPoint(x, y, 1.0) for x, y in kpts_B]
    matches_A_to_B = [cv2.DMatch(idx, idx, 0.0) for idx in range(len(kpts_A))]
    ret = cv2.drawMatches(im_A, kpts_A, im_B, kpts_B, matches_A_to_B, None)
    return ret

if __name__ == "__main__":
    img_paths = ["images/im_A.jpg", "images/im_B.jpg"]
    mynet = DeDoDeRunner_end2end('weights/dedode_end2end_1024_fp16.onnx', fp16=True)

    image_a, image_b = cv2.imread(img_paths[0]), cv2.imread(img_paths[1])
    matches_a, matches_b = mynet.detect(image_a, image_b)

    # match_img = np.hstack((image_a, image_b)) ###直接把两幅输入原图拼在一起,然后在点集里从0开始连线，是不行的。因为两幅输入原图的高宽并不是完全相同的，这就使得在两幅图里检测到的点集个数也可能不相等。因此不能直接连线的，要使用DMatch建立两个点集里的点间对应关系
    # w = image_a.shape[1]
    # for i in range(matches_a.shape[0]):
    #     cv2.line(match_img, (int(matches_a[i,0]), int(matches_a[i,1])), (int(w+matches_b[i,0]), int(w+matches_b[i,1])), (0, 255, 0), lineType=16)

    match_img = draw_matches(image_a, matches_a, image_b, matches_b)
    print('image_a.shape =',image_a.shape, 'image_b.shape =',image_b.shape, 'match_img.shape =',match_img.shape)
    cv2.namedWindow('Image matches use onnxrunime', cv2.WINDOW_NORMAL)
    cv2.imshow("Image matches use onnxrunime", match_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
