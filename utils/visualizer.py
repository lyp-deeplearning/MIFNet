# Visualization utilities
import cv2
import numpy as np



def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, rmse, H_m, name, margin=15, radius = 1):
    
    mkpt0_pre =  cv2.perspectiveTransform(mkpts0.reshape(-1, 1, 2).astype(float), H_m) 
    mkpt0_pre = mkpt0_pre.reshape(-1,2)
    H0, W0, c = image0.shape
    H1, W1, c = image1.shape
    # 输入的关键点和匹配的关键点坐标均为768 * 768， hm也是在768*768的尺度下进行计算, 图像输入是原图
    H, W = max(H0, H1), W0 + W1 + margin
    # 构建画布，把两个图像先拼接到一起
    out = 255*np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = image0
    out[:H1, W0+margin:, :] = image1
    kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
    white = (255, 255, 255)
    black = (0, 0, 0)
    # 普通的点
    vis_normal_point = True
    
    if (vis_normal_point):
        for x, y in kpts0:
            cv2.circle(out, (x, y), radius, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), radius, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), radius, black, -1,
                lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), radius, white, -1,
                lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)

    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    # memo,cf-oct是20，cf-fa是100
    for (x0, y0), (x1, y1),(x00,y00), c in zip(mkpts0, mkpts1, mkpt0_pre, color):
        c = c.tolist()
        if (np.abs(x00-x1) + np.abs(y00-y1)) < 30:
            c= (0,255,0)
        else:
            c=(0,0,255)
        # if (x0 < 100):
        #     continue
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=3, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), radius, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), radius, c, -1,
                   lineType=cv2.LINE_AA)
    return out

def make_matching_plot_fast_reverse(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, rmse, H_m, name, margin=15, radius = 1):
    
    mkpt0_pre =  cv2.perspectiveTransform(mkpts0.reshape(-1, 1, 2).astype(float) , H_m) 
    mkpt0_pre = mkpt0_pre.reshape(-1,2)
    H0, W0, c = image0.shape
    H1, W1, c = image1.shape
    # 输入的关键点和匹配的关键点坐标均为768 * 768， hm也是在768*768的尺度下进行计算, 图像输入是原图
    H, W = max(H0, H1), W0 + W1 + margin
    # 构建画布，把两个图像先拼接到一起
    out = 255*np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = image1
    out[:H1, W0+margin:, :] = image0
    kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
    white = (255, 255, 255)
    black = (0, 0, 0)
    # 普通的点
    vis_normal_point = True
    
    if (vis_normal_point):
        for x, y in kpts1:
            cv2.circle(out, (x, y), radius, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), radius, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts0:
            cv2.circle(out, (x + margin + W0, y), radius, black, -1,
                lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), radius, white, -1,
                lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)

    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    # memo,cf-oct是20，cf-fa是100
    for (x0, y0), (x1, y1),(x00,y00), c in zip(mkpts0, mkpts1, mkpt0_pre, color):
        c = c.tolist()
        if (np.abs(x00-x1) + np.abs(y00-y1)) < 30:
            c= (0,255,0)
        else:
            c=(0,0,255)
        # if (x0 < 100):
        #     continue
        cv2.line(out, (x1, y1), (x0 + margin + W0, y0),
                 color=c, thickness=3, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x1, y1), radius, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x0 + margin + W0, y0), radius, c, -1,
                   lineType=cv2.LINE_AA)
    return out



####################### 先旋转图像， 再计算关键点 #########################
def make_matching_plot_fast_reverse_rotate(random_angle, image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, rmse, H_m, name, margin=15, radius = 1):
    # 根据HM，得到预测的匹配点重映射以后的坐标, 实际img0是放在右边，img1是放在左边
    H0, W0, c = image0.shape
    H1, W1, c = image1.shape
    # warp图像，将图像进行旋转并返回rotate_matrix
    img0_rotate, rotate_matrix, H_inv_rot = rotate_image(image0, random_angle)

    H_m =  H_m @ H_inv_rot
    mkpt0_pre =  cv2.perspectiveTransform(mkpts0.reshape(-1, 1, 2).astype(float) , H_m) 
    mkpt0_pre = mkpt0_pre.reshape(-1,2)

    #rotated_keypoints_0 = rotate_keypoints(mkpts0, rotate_matrix)[:, :2]
    rotated_keypoints_0 = mkpts0
    
    # 输入的关键点和匹配的关键点坐标均为768 * 768， hm也是在768*768的尺度下进行计算, 图像输入是原图
    H, W = max(H0, H1), W0 + W1 + margin
    # 构建画布，把两个图像先拼接到一起
    out = 255*np.ones((H, W, 3), np.uint8)
    out[:H0, :W0, :] = image1
    out[:H1, W0+margin:, :] = img0_rotate
    kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
    #rotated_normals_0 = rotate_keypoints(kpts0, rotate_matrix)[:, :2]
    rotated_normals_0 = kpts0
    white = (255, 255, 255)
    black = (0, 0, 0)
    # 显示所有的点
    vis_normal_point = True
    if (vis_normal_point):
        for x, y in kpts1:
            # if (image1[y, x][0]< 5 and image1[y, x][1]< 5):
            #     continue
            cv2.circle(out, (x, y), radius, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), radius, white, -1, lineType=cv2.LINE_AA)
        for x, y in rotated_normals_0:
            cv2.circle(out, (x + margin + W0, y), radius, black, -1,
                lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), radius, white, -1,
                lineType=cv2.LINE_AA)
    rotated_keypoints_0, mkpts1 = np.round(rotated_keypoints_0).astype(int), np.round(mkpts1).astype(int)

    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    # memo,cf-oct是20，cf-fa是100
    for (x0, y0), (x1, y1),(x00,y00), c in zip(rotated_keypoints_0, mkpts1, mkpt0_pre, color):
        c = c.tolist()
        if (np.abs(x00-x1) + np.abs(y00-y1)) < 28:
            c= (0,255,0)
        else:
            c=(0,0,255)
        
        cv2.line(out, (x1, y1), (x0 + margin + W0, y0),
                 color=c, thickness=3, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x1, y1), radius, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x0 + margin + W0, y0), radius, c, -1,
                   lineType=cv2.LINE_AA)
    return out



############### 处理旋转部分的代码 #################
def rotate_image(image, angle):
    """
    旋转图像，并自动调整边界大小，以防止裁剪掉部分区域
    :param image: 输入图像
    :param angle: 旋转角度（逆时针）
    :return: 旋转后的图像
    """
    # 获取图像尺寸
    h, w = image.shape[:2]
    # 计算旋转中心
    center = (w // 2, h // 2)
    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    # 计算旋转后图像的新边界尺寸
    cos_val = np.abs(rotation_matrix[0, 0])
    sin_val = np.abs(rotation_matrix[0, 1])
    new_w = int((h * sin_val) + (w * cos_val))
    new_h = int((h * cos_val) + (w * sin_val))
      # 计算缩放比例，使旋转后的图像匹配原始大小
    scale_x = w / new_w
    scale_y = h / new_h
    scale = min(scale_x, scale_y)  # 选取最小的缩放比例，确保整个图像能完整缩放到原始尺寸
    # 更新旋转矩阵的缩放参数
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=scale)
     # 进行仿射变换 (保证输出尺寸与原图一致)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    #rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
     # 变换矩阵扩展为 3×3 单应性矩阵
    H_rot = np.vstack([rotation_matrix, [0, 0, 1]])

    # 计算逆变换矩阵（撤销旋转）
    H_inv_rot = np.linalg.inv(H_rot)
    return rotated_image, H_rot, H_inv_rot

def rotate_keypoints(keypoints, rotation_matrix):
    """
    旋转图像并计算关键点在新图像中的坐标
    :param image: 输入图像 (numpy array)
    :param keypoints: 关键点坐标列表 [(x1, y1), (x2, y2), ...]
    :param angle: 旋转角度 (默认为45度)
    :return: 旋转后的图像, 旋转后的关键点坐标
    """
    keypoints = np.array(keypoints)  # 转换为 NumPy 数组
    ones = np.ones((keypoints.shape[0], 1))  # 添加偏移量列
    keypoints_homogeneous = np.hstack([keypoints, ones])  # 齐次坐标 (x, y, 1)
    # 进行矩阵变换
    rotated_keypoints = np.dot(rotation_matrix, keypoints_homogeneous.T).T
    return  np.round(rotated_keypoints).astype(int)