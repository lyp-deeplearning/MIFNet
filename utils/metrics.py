# Evaluation metrics
import os
import cv2
import numpy as np
import pandas as pd
from utils.visualizer import rotate_keypoints
def read_sam_momo_data(gt_anno_dir):
    files_with_parent_dir = []
    # os.walk遍历路径下的所有子目录和文件
    for root, dirs, files in os.walk(gt_anno_dir):
        for file in files:
            # 获取文件所在的目录名
            parent_dir = os.path.basename(root)
            # 将文件名和它的上一级目录名添加到列表中
            files_with_parent_dir.append((file, parent_dir))

    return files_with_parent_dir

def read_cffa_file(base_dir, subfile):
    folder_path = os.path.join(base_dir, subfile)
    files = os.listdir(folder_path)
    # 初始化文件路径变量
    keypoints_path = None
    query_image_path = None
    refer_image_path = None
     # 识别关键点文件和图像文件
    
    for file in files:
        if file.endswith('.txt'):
            keypoints_path = os.path.join(folder_path, file)
        elif file.endswith('.png'):
            if 'Colour' in file:
                refer_image_path = file
            else:
                query_image_path = file
    
    return keypoints_path, refer_image_path, query_image_path 



# 怎么按照顺序读取图片对，输入是一个excel文件，返回两个图片路径以及图片的点的标注信息
def read_excel(anno_path:str):
    # 读取excel文件
    df = pd.read_csv(anno_path, header=None, skiprows=1)
    # 初始化结果列表
    result_list = []
    # 遍历每一行
    for index, row in df.iterrows():
        file_name1 = row[0]
        file_name2 = row[1]
        points = row[2:].values.reshape(12, 2)  # 将点转换为12*2的矩阵
        result_list.append([file_name1, file_name2, points])

    return result_list



class record_metrics():
    def __init__(self) -> None:
        # 需要统计的指标
        self.dice_all = 0.0
        self.image_num = 0
        self.failed = 0
        self.inaccurate = 0
        self.mae = 0
        self.mee = 0
        self.avg_dist = 0
        self.rmse = 0
        self.single_rmse = 0
        self.mae_all = 0.0
        # category: S, P, A, corresponding to Easy, Hard, Mod in paper
        self.auc_record = dict([(category, []) for category in ['S', 'P', 'A']])
    
    def update_auc(self, category, ave_dis):
        self.auc_record[category].append(ave_dis)

    def update_distance(self, fix_p, mov_p):
        dis = (fix_p - mov_p) ** 2
        dis = np.sqrt(dis[:, 0] + dis[:, 1])
        self.avg_dist = dis.mean()
        self.mae = dis.max()
        self.mee = np.median(dis)
       

        differences = fix_p - mov_p
        squared_differences = differences ** 2
        mean_squared_difference = np.mean(squared_differences)
        rmse = np.sqrt(mean_squared_difference)
        self.rmse += rmse
        self.single_rmse =  rmse
        self.mae_all += self.mae

    # 计算auc相关的指标
    # Compute AUC scores for image registration on the FIRE dataset
    def compute_auc(self, s_error, p_error, a_error):
        assert (len(s_error) == 71)  # Easy pairs
        assert (len(p_error) == 48)  # Hard pairs. Note file control_points_P37_1_2.txt is ignored
        assert (len(a_error) == 14)  # Moderate pairs

        s_error = np.array(s_error)
        p_error = np.array(p_error)
        a_error = np.array(a_error)
        # 2912的尺寸用的是25，现在是768   7
        limit = 25
        gs_error = np.zeros(limit + 1)
        gp_error = np.zeros(limit + 1)
        ga_error = np.zeros(limit + 1)

        accum_s = 0
        accum_p = 0
        accum_a = 0

        for i in range(1, limit + 1):
            gs_error[i] = np.sum(s_error < i) * 100 / len(s_error)
            gp_error[i] = np.sum(p_error < i) * 100 / len(p_error)
            ga_error[i] = np.sum(a_error < i) * 100 / len(a_error)

            accum_s = accum_s + gs_error[i]
            accum_p = accum_p + gp_error[i]
            accum_a = accum_a + ga_error[i]

        auc_s = accum_s / (limit * 100)
        auc_p = accum_p / (limit * 100)
        auc_a = accum_a / (limit * 100)
        mAUC = (auc_s + auc_p + auc_a) / 3.0
        return {'s': auc_s, 'p': auc_p, 'a': auc_a, 'mAUC': mAUC}

    def compute_auc_flori(self, s_error, p_error, a_error):
        s_error = np.array(s_error)
        p_error = np.array(p_error)
        a_error = np.array(a_error)
        limit = 100
        gs_error = np.zeros(limit + 1)
        gp_error = np.zeros(limit + 1)
        ga_error = np.zeros(limit + 1)

        accum_s = 0
        accum_p = 0
        accum_a = 0

        for i in range(1, limit + 1):
            gs_error[i] = np.sum(s_error < i) * 100 / len(s_error)
            gp_error[i] = np.sum(p_error < i) * 100 / len(p_error)
            ga_error[i] = np.sum(a_error < i) * 100 / len(a_error)

            accum_s = accum_s + gs_error[i]
            accum_p = accum_p + gp_error[i]
            accum_a = accum_a + ga_error[i]

        auc_s = accum_s / (limit * 100)
        auc_p = accum_p / (limit * 100)
        auc_a = accum_a / (limit * 100)
        mAUC = (auc_s + auc_p + auc_a) / 3.0
        return {'s': auc_s, 'p': auc_p, 'a': auc_a, 'mAUC': mAUC}

        
    def print_log_info(self, auc_type="fire"):
        # 输出总共的点的误差
        print('-'*40)
        print(f"Failed:{'%.2f' % (100*self.failed/self.image_num)}%, Inaccurate:{'%.2f' % (100*self.inaccurate/self.image_num)}%, "
                f"Acceptable:{'%.2f' % (100*(self.image_num-self.inaccurate-self.failed)/self.image_num)}%")
        print('-'*40)
        if (auc_type =="fire"):
            auc = self.compute_auc(self.auc_record['S'], self.auc_record['P'], self.auc_record['A'])
        else:
            auc = self.compute_auc_flori(self.auc_record['S'], self.auc_record['P'], self.auc_record['A'])
        print('S: %.3f, P: %.3f, A: %.3f, mAUC: %.3f' % (auc['s'], auc['p'], auc['a'], auc['mAUC']))
        txt_line2='S: %.3f, P: %.3f, A: %.3f, mAUC: %.3f' % (auc['s'], auc['p'], auc['a'], auc['mAUC'])
        # with open('/home/yepeng_liu/code_base/awesome-diffusion/dift/retina_fire_test/record.txt', 'a') as file:
        # # 写入内容并换行
        #     file.write(txt_line2 + '\n')


class cal_auc_sam_mm:
    def __init__(self, mae=100, mee=60, scale_x=1.0, scale_y=1.0) -> None:
        self.metrics_predict = record_metrics_sam_mm()
        self.mae = mae
        self.mee = mee
        self.scale_x = scale_x
        self.scale_y = scale_y

    def update(self, gt_file, homography_matrix, category):
        self.metrics_predict.image_num += 1

        points_gd = np.loadtxt(gt_file, skiprows=1)
        fix = np.zeros([len(points_gd-1), 2])
        mov = np.zeros([len(points_gd-1), 2])
        fix[:, 0] = points_gd[:, 0] * self.scale_x #dst对应的是refer 1 fix，raw对应的是query 2 mov
        fix[:, 1] = points_gd[:, 1] * self.scale_y
        mov[:, 0] = points_gd[:, 2] * self.scale_x
        mov[:, 1] = points_gd[:, 3] * self.scale_y
        
        H_m_gt, mask = cv2.findHomography(mov, fix, cv2.LMEDS)#cv2.RANSAC,cv2.LMEDS
        
        if (homography_matrix.shape[0] == 1):
            self.metrics_predict.failed += 1
            return False,H_m_gt
        
        dst_pred = cv2.perspectiveTransform(mov.reshape(-1, 1, 2), homography_matrix)
        
        dst_pred = dst_pred.reshape(-1,2)
        rmse_success = self.metrics_predict.update_distance(fix_p=fix, mov_p=dst_pred)
        #rmse_success = self.metrics_predict.update_distance(fix_p=fix, mov_p=mov)
        if (rmse_success == False):
            return False,H_m_gt
        # 50,20
        if self.metrics_predict.mae > self.mae or self.metrics_predict.mee > self.mee:
            self.metrics_predict.inaccurate += 1
        
        print("[info] failed:[%d],image_num:[%d],inaccurate:[%d], mae:[%f], mee:[%f]",self.metrics_predict.failed, \
              self.metrics_predict.image_num, self.metrics_predict.inaccurate, self.metrics_predict.mae, self.metrics_predict.mee)
        if (len(category.split("_")) == 2):
            category_new = category.split("_")[0]
        else:
            category_new = category
        self.metrics_predict.update_auc(category_new, self.metrics_predict.avg_dist)
        return True, H_m_gt
    
    def update_rotate(self, gt_file, homography_matrix, category, H_rot_):
        self.metrics_predict.image_num += 1

        points_gd = np.loadtxt(gt_file, skiprows=1)
        fix = np.zeros([len(points_gd-1), 2])
        mov = np.zeros([len(points_gd-1), 2])
        fix[:, 0] = points_gd[:, 0] * self.scale_x #dst对应的是refer 1 fix，raw对应的是query 2 mov
        fix[:, 1] = points_gd[:, 1] * self.scale_y
        mov[:, 0] = points_gd[:, 2] * self.scale_x
        mov[:, 1] = points_gd[:, 3] * self.scale_y
        
        H_m_gt, mask = cv2.findHomography(mov, fix, cv2.LMEDS)#cv2.RANSAC,cv2.LMEDS
        # 要重新旋转一下
        mov_rotate =  rotate_keypoints(mov, H_rot_)[:, :2].astype(np.float64)
        
        if (homography_matrix.shape[0] == 1):
            self.metrics_predict.failed += 1
            return False,H_m_gt
        
        dst_pred = cv2.perspectiveTransform(mov_rotate.reshape(-1, 1, 2), homography_matrix)
        
        dst_pred = dst_pred.reshape(-1,2)
        rmse_success = self.metrics_predict.update_distance(fix_p=fix, mov_p=dst_pred)
        #rmse_success = self.metrics_predict.update_distance(fix_p=fix, mov_p=mov)
        if (rmse_success == False):
            return False,H_m_gt
        # 50,20
        if self.metrics_predict.mae > self.mae or self.metrics_predict.mee > self.mee:
            self.metrics_predict.inaccurate += 1
        
        print("[info] failed:[%d],image_num:[%d],inaccurate:[%d], mae:[%f], mee:[%f]",self.metrics_predict.failed, \
              self.metrics_predict.image_num, self.metrics_predict.inaccurate, self.metrics_predict.mae, self.metrics_predict.mee)
       
        if (len(category.split("_")) == 2):
            category_new = category.split("_")[0]
        else:
            category_new = category
        self.metrics_predict.update_auc(category_new, self.metrics_predict.avg_dist)
        return True, H_m_gt
    
class cal_auc_cfoct:
    def __init__(self, mae=100, mee=60) -> None:
        self.metrics_predict = record_metrics_sam_mm()
        self.mae = mae
        self.mee = mee

    def update(self, gt_file, homography_matrix, category,q_w, q_h,r_w, r_h,\
               r1,r2,q1,q2):
        self.metrics_predict.image_num += 1

        points_gd = np.loadtxt(gt_file)
        fix = np.zeros([len(points_gd), 2])
        mov = np.zeros([len(points_gd), 2])
        #dst对应的是refer 1 fix，raw对应的是query 2 mov
        fix[:, 0] = (points_gd[:, 0] / r_w) * r1
        fix[:, 1] = (points_gd[:, 1] / r_h) * r2
        mov[:, 0] = (points_gd[:, 2] / q_w) * q1 
        mov[:, 1] = (points_gd[:, 3]  / q_h) * q2
       
        H_m_gt, mask = cv2.findHomography(mov, fix, cv2.LMEDS)#cv2.RANSAC,cv2.LMEDS
        
        if (homography_matrix.shape[0] == 1):
            self.metrics_predict.failed += 1
            return False,H_m_gt
        
        dst_pred = cv2.perspectiveTransform(mov.reshape(-1, 1, 2), homography_matrix)
        dst_pred = dst_pred.reshape(-1,2)
        rmse_success = self.metrics_predict.update_distance(fix_p=fix, mov_p=dst_pred)
        # 直接计算配准前误差
        #rmse_success = self.metrics_predict.update_distance(fix_p=fix, mov_p=mov)
       
        if (rmse_success == False):
            return False,H_m_gt
        # 50,20
        if self.metrics_predict.mae > self.mae or self.metrics_predict.mee > self.mee:
            self.metrics_predict.inaccurate += 1
        
        print("[info] failed:[%d],image_num:[%d],inaccurate:[%d], mae:[%f], mee:[%f]",self.metrics_predict.failed, \
              self.metrics_predict.image_num, self.metrics_predict.inaccurate, self.metrics_predict.mae, self.metrics_predict.mee)
        if (len(category.split("_")) == 2):
            category_new = category.split("_")[0]
        else:
            category_new = category
        self.metrics_predict.update_auc(category_new, self.metrics_predict.avg_dist)
        return True, H_m_gt
    
    def update_rotate(self, gt_file, homography_matrix, category,q_w, q_h,r_w, r_h,\
               r1,r2,q1,q2, H_rot_):
        self.metrics_predict.image_num += 1

        points_gd = np.loadtxt(gt_file)
        fix = np.zeros([len(points_gd), 2])
        mov = np.zeros([len(points_gd), 2])
        #dst对应的是refer 1 fix，raw对应的是query 2 mov
        fix[:, 0] = (points_gd[:, 0] / r_w) * r1
        fix[:, 1] = (points_gd[:, 1] / r_h) * r2
        mov[:, 0] = (points_gd[:, 2] / q_w) * q1 
        mov[:, 1] = (points_gd[:, 3]  / q_h) * q2
        
        H_m_gt, mask = cv2.findHomography(mov, fix, cv2.LMEDS)#cv2.RANSAC,cv2.LMEDS
        # 要重新旋转一下
        mov_rotate =  rotate_keypoints(mov, H_rot_)[:, :2].astype(np.float64)
        
        if (homography_matrix.shape[0] == 1):
            self.metrics_predict.failed += 1
            return False,H_m_gt
        
        dst_pred = cv2.perspectiveTransform(mov_rotate.reshape(-1, 1, 2), homography_matrix)
        dst_pred = dst_pred.reshape(-1,2)
        rmse_success = self.metrics_predict.update_distance(fix_p=fix, mov_p=dst_pred)
       
        if (rmse_success == False):
            return False,H_m_gt
        # 50,20
        if self.metrics_predict.mae > self.mae or self.metrics_predict.mee > self.mee:
            self.metrics_predict.inaccurate += 1
        
        print("[info] failed:[%d],image_num:[%d],inaccurate:[%d], mae:[%f], mee:[%f]",self.metrics_predict.failed, \
              self.metrics_predict.image_num, self.metrics_predict.inaccurate, self.metrics_predict.mae, self.metrics_predict.mee)
        if (len(category.split("_")) == 2):
            category_new = category.split("_")[0]
        else:
            category_new = category
        self.metrics_predict.update_auc(category_new, self.metrics_predict.avg_dist)
        return True, H_m_gt

class cal_auc_memo:
    def __init__(self, mae=100, mee=60) -> None:
        self.metrics_predict = record_metrics_sam_mm()
        self.mae = mae
        self.mee = mee

    def update(self, pt_gt, homography_matrix, category, q_w, q_h, r_w, r_h,\
               r1, r2, q1, q2):
        self.metrics_predict.image_num += 1
        
        mov = np.zeros([6, 2]) 
        fix = np.zeros([6, 2])
        mov = pt_gt[:6 , : ].astype(np.float32) 
        fix = pt_gt[6: , : ].astype(np.float32) 
        
        fix[:, 0] = (fix[:, 0] / r_w) * r1
        fix[:, 1] = (fix[:, 1] / r_h) * r2
        mov[:, 0] = (mov[:, 0] / q_w) * q1 
        mov[:, 1] = (mov[:, 1]  / q_h) * q2
        
        H_m_gt, mask = cv2.findHomography(mov, fix, cv2.LMEDS)#cv2.RANSAC,cv2.LMEDS
        

        if (homography_matrix.shape[0] == 1):
            self.metrics_predict.failed += 1
            return False,H_m_gt
        
        dst_pred = cv2.perspectiveTransform(mov.reshape(-1, 1, 2), homography_matrix)
        dst_pred = dst_pred.reshape(-1,2)
        rmse_success = self.metrics_predict.update_distance(fix_p=fix, mov_p=dst_pred)
        # 直接计算配准前误差
        #rmse_success = self.metrics_predict.update_distance(fix_p=fix, mov_p=mov)

        if (rmse_success == False):
            return False,H_m_gt
        # 50,20
        if self.metrics_predict.mae > self.mae or self.metrics_predict.mee > self.mee:
            self.metrics_predict.inaccurate += 1
        
        print("[info] failed:[%d],image_num:[%d],inaccurate:[%d], mae:[%f], mee:[%f]",self.metrics_predict.failed, \
              self.metrics_predict.image_num, self.metrics_predict.inaccurate, self.metrics_predict.mae, self.metrics_predict.mee)
        if (len(category.split("_")) == 2):
            category_new = category.split("_")[0]
        else:
            category_new = category
        self.metrics_predict.update_auc(category_new, self.metrics_predict.avg_dist)
        return True, H_m_gt
    
    def update_rotate(self, pt_gt, homography_matrix, category, q_w, q_h, r_w, r_h,\
               r1, r2, q1, q2, H_rot_):
        self.metrics_predict.image_num += 1
        
        mov = np.zeros([6, 2]) 
        fix = np.zeros([6, 2])
        mov = pt_gt[:6 , : ].astype(np.float32) 
        fix = pt_gt[6: , : ].astype(np.float32) 
        
        fix[:, 0] = (fix[:, 0] / r_w) * r1
        fix[:, 1] = (fix[:, 1] / r_h) * r2
        mov[:, 0] = (mov[:, 0] / q_w) * q1 
        mov[:, 1] = (mov[:, 1]  / q_h) * q2
        
        H_m_gt, mask = cv2.findHomography(mov, fix, cv2.LMEDS)#cv2.RANSAC,cv2.LMEDS
        # 要重新旋转一下
        mov_rotate =  rotate_keypoints(mov, H_rot_)[:, :2].astype(np.float64)
        

        if (homography_matrix.shape[0] == 1):
            self.metrics_predict.failed += 1
            return False,H_m_gt
        
        dst_pred = cv2.perspectiveTransform(mov_rotate.reshape(-1, 1, 2), homography_matrix)
        dst_pred = dst_pred.reshape(-1,2)
        rmse_success = self.metrics_predict.update_distance(fix_p=fix, mov_p=dst_pred)
        # 直接计算配准前误差
        #rmse_success = self.metrics_predict.update_distance(fix_p=fix, mov_p=mov)

        if (rmse_success == False):
            return False,H_m_gt
        # 50,20
        if self.metrics_predict.mae > self.mae or self.metrics_predict.mee > self.mee:
            self.metrics_predict.inaccurate += 1
        
        print("[info] failed:[%d],image_num:[%d],inaccurate:[%d], mae:[%f], mee:[%f]",self.metrics_predict.failed, \
              self.metrics_predict.image_num, self.metrics_predict.inaccurate, self.metrics_predict.mae, self.metrics_predict.mee)
        if (len(category.split("_")) == 2):
            category_new = category.split("_")[0]
        else:
            category_new = category
        self.metrics_predict.update_auc(category_new, self.metrics_predict.avg_dist)
        return True, H_m_gt
    
    
class record_metrics_sam_mm():
    def __init__(self) -> None:
        # 需要统计的指标
        self.dice_all = 0.0
        self.image_num = 0
        self.failed = 0
        self.inaccurate = 0
        self.mae = 0
        self.mee = 0
        self.avg_dist = 0
        self.rmse = 0.0
        self.mae_all = 0.0
        self.single_rmse = 0
        # category: S, P, A, corresponding to Easy, Hard, Mod in paper
        self.auc_record = dict([(category, []) for category in ['easy', 'hard']])
    
    def update_auc(self, category, ave_dis):
        self.auc_record[category].append(ave_dis)

    def update_distance(self, fix_p, mov_p):
        differences = fix_p - mov_p
        squared_differences = differences ** 2
        mean_squared_difference = np.mean(squared_differences)
        rmse = round(np.sqrt(mean_squared_difference) , 2)
        self.single_rmse =  rmse
        if (rmse > 200):
            self.failed += 1
            return False

        dis = (fix_p - mov_p) ** 2
        dis = np.sqrt(dis[:, 0] + dis[:, 1])
        self.avg_dist = round(dis.mean(), 2)
        self.mae = round(dis.max(), 2)
        self.mee = round(np.median(dis), 2)
        
        
        self.rmse += self.single_rmse 
        self.mae_all += self.mae

    # 计算auc相关的指标
    # Compute AUC scores for image registration on the FIRE dataset
    def compute_auc(self, easy_error, hard_error):
        easy_error = np.array(easy_error)
        hard_error = np.array(hard_error)
        # 2912的尺寸用的是25，现在是768   7
        limit = 40
        gs_error = np.zeros(limit + 1)
        gp_error = np.zeros(limit + 1)
        accum_s = 0
        accum_p = 0
        for i in range(1, limit + 1):
            gs_error[i] = np.sum(easy_error < i) * 100 / len(easy_error)
            gp_error[i] = np.sum(hard_error < i) * 100 / len(hard_error)
            

            accum_s = accum_s + gs_error[i]
            accum_p = accum_p + gp_error[i]
            

        auc_s = accum_s / (limit * 100)
        auc_p = accum_p / (limit * 100)
        
        mAUC = (auc_s + auc_p ) / 2.0
        return {'easy': auc_s, 'hard': auc_p, 'mAUC': mAUC}

    def print_log_info(self):
        # 输出总共的点的误差
        print('-'*40)
        print(f"Failed:{'%.2f' % (100*self.failed/self.image_num)}%, Inaccurate:{'%.2f' % (100*self.inaccurate/self.image_num)}%, "
                f"Acceptable:{'%.2f' % (100*(self.image_num-self.inaccurate-self.failed)/self.image_num)}%")
        print('-'*40)
        auc = self.compute_auc(self.auc_record['easy'], self.auc_record['hard']) 
        print('easy: %.3f, hard: %.3f,  mAUC: %.3f' % (auc['easy'], auc['hard'], auc['mAUC']))
        print('-'*40)
        return (100*(self.image_num-self.inaccurate-self.failed)/self.image_num)
    
