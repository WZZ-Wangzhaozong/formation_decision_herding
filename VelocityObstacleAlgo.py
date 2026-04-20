import numpy as np


class VelocityObstacle:
    def __init__(self):
        pass


def static_orca(states, v_pref, obs_circle, obs_poly, obs_dis, t_vo, v_max, r_adjust):
    """
    Description:
    ------------
        静态环境下的orca算法，用于计算最优速度
    Parameters:
    -----------
        states: 当前状态
        v_pref: 期望速度
        obs_circle: 圆形障碍物, list of [x, y, r]
        obs_poly: 多边形障碍物, list of [x1, y1, x2, y2, ..., xn, yn]
        obs_dis: 安全距离
        t_vo: 预测时间，速度障碍考虑的时间窗口
        v_max: 最大速度，自己门回头自大速度
        r_adjust: 调整半径，考虑避障的缓冲距离
    """
    v_cmd = v_pref
    flag_collide = False

    # 圆形障碍处理
    v_ref_circ = v_pref
    num_obs_circ = len(obs_circle)
    num_near_circ = num_obs_circ
    ptx_circ, pty_circ, dirx_circ, diry_circ = [], [], [], []

    if num_obs_circ > 0:
        obs_circle_pos = obs_circle[:, 0:2]  # 圆心坐标
        obs_circ_r = obs_circle[:, 2].reshape(-1)  # 半径
        displa_circ_center = obs_circle_pos - states[0:2]  # 到各圆心的位移
        dis_circ_center = np.linalg.norm(displa_circ_center, axis=1)  # 到各圆心的距离
        dis_circ = dis_circ_center - obs_circ_r  # 智能体到各障碍的距离
        collide_circ = dis_circ < obs_dis  # 是否碰撞，找出碰撞的障碍物
        num_collide_circ = np.sum(collide_circ)

        if num_collide_circ > 0:
            flag_collide = True
            return v_cmd, flag_collide  # 碰撞则返回，不再计算，直接返回零速度

        near_circ_index = dis_circ < obs_dis + t_vo * v_max  # 找出预测时间内可能碰撞的障碍物
        num_near_circ = np.sum(near_circ_index)

        if num_near_circ > 0:  # 如果有可能碰撞的障碍物
            near_displa_center = displa_circ_center[near_circ_index, :]
            near_dis_center = dis_circ_center[near_circ_index]
            near_obs_circ_r = obs_circ_r[near_circ_index]
            # near_vel_unit_circle = (v_ref_circ - near_displa_center / t_vo) / np.maximum(np.linalg.norm(v_ref_circ - near_displa_center / t_vo, axis=1), 1E-6))  # 最小无碰撞改变量 (单位方向向量)
            near_vel_unit_circle = ((v_ref_circ - near_displa_center / t_vo)
                                    / np.maximum(np.linalg.norm(v_ref_circ - near_displa_center / t_vo, axis=1), 0.0001))

            pt_circ = np.zeros((num_near_circ, 2))
            ptx_circ, pty_circ = np.zeros(num_near_circ), np.zeros(num_near_circ)
            dirx_circ, diry_circ = np.zeros(num_near_circ), np.zeros(num_near_circ)

            angle_unit = np.arctan2(near_vel_unit_circle[:, 1], near_vel_unit_circle[:, 0])
            angle_center = np.arctan2(near_displa_center[:, 1], near_displa_center[:, 0])
            angle_left = np.arcsin((near_obs_circ_r + obs_dis) / near_dis_center) + angle_center
            angle_right = -np.arcsin((near_obs_circ_r + obs_dis) / near_dis_center) + angle_center
            angle_left_in = angle_left + np.pi / 2
            angle_right_in = angle_right - np.pi / 2

            side_center = is_left_side(angle_unit, angle_center)
            side_left = is_left_side(angle_unit, angle_left_in)
            side_right = is_left_side(angle_unit, angle_right_in)

            index_front = np.logical_and(~side_right, side_left)
            index_left = np.logical_and(side_center, ~side_left)
            index_right = np.logical_and(side_right, ~side_center)

            if np.sum(index_front) > 0:
                pt_circ[index_front] = (near_displa_center[index_front] + near_vel_unit_circle[index_front] * (
                                                   near_obs_circ_r[index_front] + obs_dis)) / t_vo
                ptx_circ[index_front] = pt_circ[index_front, 0]
                pty_circ[index_front] = pt_circ[index_front, 1]
                dirx_circ[index_front] = near_vel_unit_circle[index_front, 1]
                diry_circ[index_front] = -near_vel_unit_circle[index_front, 0]

            if np.sum(index_left) > 0:
                dirx_circ[index_left] = np.cos(angle_left[index_left])
                diry_circ[index_left] = np.sin(angle_left[index_left])

            if np.sum(index_right) > 0:
                dirx_circ[index_right] = -np.cos(angle_right[index_right])
                diry_circ[index_right] = -np.sin(angle_right[index_right])

    # 多边形障碍
    ptx_poly = np.array([])
    pty_poly = np.array([])
    dirx_poly = np.array([])
    diry_poly = np.array([])
    num_poly_valid = 0
    num_obs_poly = len(obs_poly)
    num_near_poly = num_obs_poly
    # 计算需要考虑的多边形障碍物
    if num_obs_poly > 0:
        dis_poly = np.zeros(num_obs_poly)
        displa_poly = np.zeros((num_obs_poly, 2))

        for i in range(num_obs_poly):
            A = obs_poly[i]
            B = np.roll(A, shift=-1, axis=0)
            dis_poly[i], displa_poly[i] = point2segments(states[0:2], A, B)

        index_collide_poly = dis_poly - obs_dis < 0

        if np.sum(index_collide_poly) > 0:
            flag_collide = True
            return v_cmd, flag_collide

        index_near_poly = (dis_poly - obs_dis - r_adjust) / t_vo <= v_max
        num_near_poly = np.sum(index_near_poly)

        # 计算个障碍区域的角度边界线

        if num_near_poly > 0:
            obs_near_poly = np.array(obs_poly)[index_near_poly]
            dis_near_poly = np.array(dis_poly)[index_near_poly]

            num_poly_valid = num_near_poly
            num_poly_advoid = 0
            angle_near_poly = np.zeros((num_near_poly, 2))
            for i in range(num_near_poly):
                points = obs_near_poly[i]
                points_rel = points - states[0:2]
                angle_center = np.arctan2(points_rel[:, 1], points_rel[:, 0])
                points_norm = np.linalg.norm(points_rel, axis=1)
                angle_delta = np.arcsin(obs_dis / points_norm)
                # angle_center = angle_center.reshape(-1, 1)
                # angle_delta = angle_delta.reshape(-1, 1)
                angle_tangent = np.hstack([angle_center - angle_delta, angle_center + angle_delta])
                angle_tangent = limit_angle(angle_tangent)

                angle_max, index_max = np.max(angle_tangent), np.argmax(angle_tangent)
                angle_min, index_min = np.min(angle_tangent), np.argmin(angle_tangent)

                if angle_max - angle_min <= np.pi:
                    angle_near_poly[i, 0] = angle_max
                    angle_near_poly[i, 1] = angle_min
                else:
                    index_positive = angle_tangent >= 0
                    index_negtive = angle_tangent < 0
                    angle_near_poly[i, 1] = np.min(angle_tangent[index_positive])
                    angle_near_poly[i, 0] = np.max(angle_tangent[index_negtive])

            angle_margin = np.radians(2)  # 家都缓冲范围，2 degree
            pref_angle = np.arctan2(v_pref[1], v_pref[0])
            delta_left = abs(limit_angle(angle_near_poly[:, 0] - pref_angle))
            delta_right = abs(limit_angle(angle_near_poly[:, 1] - pref_angle))
            delta_poly = abs(limit_angle(angle_near_poly[:, 0] - angle_near_poly[:, 1]))
            index_poly_valid = (delta_left <= delta_poly + angle_margin * 3) & (
                    delta_right <= delta_poly + angle_margin * 3)
            num_poly_valid = np.sum(index_poly_valid)

            # 根据不同的距离，选择不同的策略
            if num_poly_valid > 0:
                angle_poly_valid = angle_near_poly[index_poly_valid]
                delta_left_valid = delta_left[index_poly_valid]
                delta_right_valid = delta_right[index_poly_valid]
                dis_poly_valid = dis_near_poly[index_poly_valid]
                # 将valid分为avoid和adjust两类
                index_poly_avoid = dis_poly_valid < obs_dis + t_vo * v_max  # 必须避障的障碍物
                index_poly_adjust = np.array([not idx for idx in index_poly_avoid])  # 需要调整方向的障碍物
                num_poly_avoid = np.sum(index_poly_avoid)
                num_poly_adjust = np.sum(index_poly_adjust)
                angle_poly_avoid = angle_poly_valid[index_poly_avoid]
                angle_poly_adjust = angle_poly_valid[index_poly_adjust]
                # 处理必须避障的障碍物
                if num_poly_avoid > 0:
                    delta_left_avoid = delta_left_valid[index_poly_avoid]
                    delta_right_avoid = delta_right_valid[index_poly_avoid]
                    index_left_side_avoid = np.array([np.abs(left) < np.abs(right) for left, right in zip(delta_left_avoid, delta_right_avoid)])
                    # index_left_side_avoid = np.array([np.abs(left)<np.abs(right) for left, right in zip(delta_left_avoid, delta_right_avoid])
                    index_right_side_avoid = np.array([not idx for idx in index_left_side_avoid])
                    ptx_poly = np.zeros(num_poly_avoid)
                    pty_poly = np.zeros(num_poly_avoid)
                    angle_poly_avoid[:, 0] = limit_angle(angle_poly_avoid[:, 0] + angle_margin)  # 增加缓冲范围(角度余量)
                    angle_poly_avoid[:, 1] = limit_angle(angle_poly_avoid[:, 1] - angle_margin)
                    dirx_poly = (np.cos(angle_poly_avoid[:, 0]) * index_left_side_avoid
                                 - np.cos(angle_poly_avoid[:, 1]) * index_right_side_avoid)
                    diry_poly = (np.sin(angle_poly_avoid[:, 0]) * index_left_side_avoid
                                 - np.sin(angle_poly_avoid[:, 1]) * index_right_side_avoid)
                if num_poly_adjust > 0:  # 仅当没有必须规避的障碍物时，才考虑调整方向的障碍物
                    # 只根据最近障碍调整方向
                    delta_left_adjust = delta_left_valid[index_poly_adjust]
                    delta_right_adjust = delta_right_valid[index_poly_adjust]
                    index_left_side_adjust = np.abs(delta_left_adjust) < np.abs(delta_right_adjust)
                    index_right_side_adjust = np.array([not idx for idx in index_left_side_adjust])
                    ptx_adjust = np.zeros(num_poly_adjust)
                    pty_adjust = np.zeros(num_poly_adjust)
                    angle_poly_adjust[:, 0] = limit_angle(angle_poly_adjust[:, 0] + angle_margin)  # 增加缓冲范围（角度余量）
                    angle_poly_adjust[:, 1] = limit_angle(angle_poly_adjust[:, 1] - angle_margin)
                    # 根据距离缩短角度偏差
                    dis_poly_adjust = dis_poly_valid[index_poly_adjust]
                    k_adjust = (dis_poly_adjust - obs_dis - t_vo * v_max) / r_adjust
                    angle_poly_adjust = limit_angle(limit_angle(angle_poly_adjust - pref_angle)*(1 - k_adjust) + pref_angle)
                    dirx_adjust = (np.cos(angle_poly_adjust[:, 0]) * index_left_side_adjust
                                   - np.cos(angle_poly_adjust[:, 1]) * index_right_side_adjust)
                    diry_adjust = (np.sin(angle_poly_adjust[:, 0]) * index_left_side_adjust
                                   - np.sin(angle_poly_adjust[:, 1]) * index_right_side_adjust)
    # 线性规划求解最优解
    ptx = np.array([])
    pty = np.array([])
    dirx = np.array([])
    diry = np.array([])

    if num_obs_circ > 0 and num_near_circ > 0:
        ptx = np.hstack([ptx, ptx_circ])
        pty = np.hstack([pty, pty_circ])
        dirx = np.hstack([dirx, dirx_circ])
        diry = np.hstack([diry, diry_circ])
    if num_obs_poly > 0 and num_poly_valid > 0:
        if num_poly_avoid > 0:
            ptx = np.hstack([ptx, ptx_poly])
            pty = np.hstack([pty, pty_poly])
            dirx = np.hstack([dirx, dirx_poly])
            diry = np.hstack([diry, diry_poly])
        if num_poly_adjust > 0:
            ptx = np.hstack([ptx, ptx_adjust])
            pty = np.hstack([pty, pty_adjust])
            dirx = np.hstack([dirx, dirx_adjust])
            diry = np.hstack([diry, diry_adjust])
    if len(ptx) > 0:
        v_cmd, is_success = cal_opt_coll_free_vel(v_pref, v_max, ptx, pty, dirx, diry)
        if not is_success:
            flag_collide = True


    return v_cmd, flag_collide


def point2segments(point, A, B):
    num = A.shape[0]
    dis = np.zeros(num)
    nearest_point = np.zeros((num, 2))
    for i in range(num):
        dis[i], nearest_point[i] = point2segment(point, A[i, :], B[i, :])
    dis_min = np.min(dis)
    nearest_point_min = nearest_point[np.argmin(dis)]
    return dis_min, nearest_point_min


def point2segment(point, a, b):
    d = b - a
    r = point - a
    t = np.dot(r, d) / np.dot(d, d)

    if t < 0:
        t = 0
    elif t > 1:
        t = 1

    nearest_point = a + t * d
    dis = np.linalg.norm(nearest_point - point)

    return dis, nearest_point


def is_left_side(angle_unit, angle_in):
    return np.mod(angle_in - angle_unit + np.pi, 2 * np.pi) < np.pi


def limit_angle(angle):
    return np.mod(angle + np.pi, 2 * np.pi) - np.pi


def cal_opt_coll_free_vel(v_pref, v_max, ptx, pty, dirx, diry):
    """
    计算无碰撞的最优速度
    """
    amp = 0.0210
    norm = amp * np.random.rand()
    angle = 2.0 * np.pi * np.random.rand()
    disturb = norm * np.array([np.cos(angle), np.sin(angle)])
    v_pref = v_pref + disturb

    num_vo = ptx.shape[0]
    line_success, v_cmd = linear_program_2(0, num_vo, v_max, v_pref, ptx, pty, dirx, diry)

    if line_success < num_vo:
        is_success = False
    else:
        is_success = True

    return v_cmd, is_success


def linear_program_2(dir_opt, vo_num, max_vel, vpre_set, ptx_set, pty_set, dirx_set, diry_set):
    if dir_opt:
        vpre_norm = np.linalg.norm(vpre_set)
        vpre_norm[vpre_norm < 0.0001] = 1  # 避免奇异值
        vpre_set = vpre_set/vpre_norm
        new_vel = max_vel*vpre_set
    elif np.linalg.norm(vpre_set) > max_vel:
        new_vel = vpre_set/np.linalg.norm(vpre_set)*max_vel
    else:
        new_vel = vpre_set
    # 分别对智能体的每个速度障碍线进行求解
    for line in range(0, vo_num):
        limit_det = dirx_set[line] * (pty_set[line] - new_vel[1]) - diry_set[line] * (ptx_set[line] - new_vel[0])
        pre_result = new_vel
        # 计算期望速度不满足速度障碍约束的智能体和障碍线
        if limit_det > 0:
            prog_flag, new_vel = linear_program_1(line, pre_result, dir_opt, max_vel, vpre_set,
                                                  ptx_set[:line+1], pty_set[:line+1], dirx_set[:line+1], diry_set[:line+1])
            # 规划不成功,返回当前速度障碍线索引
            if not prog_flag:
                new_vel = pre_result
                line -= 1
                break

    line_fail = line + 1
    return line_fail, new_vel

def linear_program_1(line, pre_result, dir_opt, max_vel, vpre_set, ptx_set, pty_set, dirx_set, diry_set):
    epsilon = 0.00001
    # 判断最大速度是否满足避碰条件
    # 计算障碍线向量顶点在障碍线方向上的投影距离
    dot_pro = ptx_set[line] * dirx_set[line] + pty_set[line] * diry_set[line]
    # 计算速度障碍线与最大速度约束圆相交线的半弦长的平方
    disc_sq = max_vel**2 - (ptx_set[line]**2 + pty_set[line]**2 - dot_pro**2)
    # 最大速度也无法避免碰撞，规划失败
    if disc_sq < 0:
        prog_flag = False
        new_vel = pre_result
        return prog_flag, new_vel
    # 计算约束区间 %%%%%%%%%%%%%%%%%%%%%
    # 计算速度障碍线与最大速度约束圆相交线的半弦长
    disc_value = np.sqrt(disc_sq)
    # 计算速度障碍线与最大速度约束圆左焦点到速度障碍线顶点的距离
    left_value = -dot_pro - disc_value
    # 计算速度障碍线与最大速度约束圆右焦点到速度障碍线顶点的距离
    right_value = -dot_pro + disc_value
    # 整合当前速度障碍线与之前速度障碍线的约束区间
    pre_num = max(0, line - 1)
    for i in range(1, pre_num + 1):
        # 计算当前速度障碍线与此障碍线之前速度障碍线i的向量叉积
        deno_value = dirx_set[line] * diry_set[i - 1] - diry_set[line - 1] * dirx_set[i - 1]
        temp_x = ptx_set[line - 1] - ptx_set[i - 1]
        temp_y = pty_set[line - 1] - pty_set[i - 1]
        nume_value = dirx_set[i - 1] * temp_y - diry_set[i - 1] * temp_x
        # 判断速度障碍线line是否与line之前的速度障碍线平行
        if abs(deno_value) < epsilon:
            # 速度障碍线line与line之前的速度障碍线之间没有交集,故规划失败
            if nume_value < 0:
                prog_flag = False
                new_vel = pre_result
                return prog_flag, new_vel
            # 速度障碍线line与line之前的速度障碍线存在重叠,故忽略该障碍线
            else:
                continue
        # 当速度障碍线line有效时,重新计算约束区间
        # 计算障碍线line顶点到line之前障碍线顶点的距离,采用相似原理,斜边长度为单位1
        ratio = nume_value / deno_value
        # line之前障碍线在障碍线line顶点的右边限制了障碍线line
        if deno_value >= 0:
            right_value = min(ratio, right_value)
        # line之前障碍线在障碍线line顶点的左边限制了障碍线lin
        else:
            left_value = max(ratio, left_value)

        # 从原点(智能体A)向速度障碍线看过去,障碍线顶点右侧距离为正,左侧为负
        # 若不满足,则当前速度障碍线lineNo与该障碍线i之间不存在交集
        if left_value > right_value:
            prog_flag = False
            new_vel = pre_result
            return prog_flag, new_vel
    # 根据约束求解线性规划结果 %%%%%%%%%%
    # 对计算结果进行速度方向优化,直接取最优端点作为输出结果
    if dir_opt:
        temp = vpre_set[0] * dirx_set[line] + vpre_set[1] * diry_set[line]
        # 期望速度在速度障碍线上的投影在障碍线顶点的右侧时,取障碍线与最大速度约束的右交点作为输出结果
        if temp > 0:
            new_vel = np.array([ptx_set[line] + right_value * dirx_set[line],
                                pty_set[line] + right_value * diry_set[line]])
        # 期望速度在速度障碍线上的投影在障碍线顶点的左侧时, 取障碍线与最大速度约束的左交点作为输出结果
        else:
            new_vel = np.array([ptx_set[line] + left_value * dirx_set[line],
                                pty_set[line] + left_value * diry_set[line]])
    else:
        # 计算期望速度相对于障碍线顶点到速度障碍线的距离
        temp = dirx_set[line] * (vpre_set[0] - ptx_set[line]) + \
               diry_set[line] * (vpre_set[1] - pty_set[line])
        # 当距离投影点在障碍线与最大速度约束交点的左侧时,则取左交点作为输出结果
        if temp < left_value:
            new_vel = np.array([ptx_set[line] + left_value * dirx_set[line],
                                pty_set[line] + left_value * diry_set[line]])
        # 当距离投影点在障碍线与最大速度约束交点的右侧时,则取右交点作为输出结果
        elif temp > right_value:
            new_vel = np.array([ptx_set[line] + right_value * dirx_set[line],
                                pty_set[line] + right_value * diry_set[line]])
        # 当距离投影点在障碍线与最大速度约束交点的弦长上时,取该投影点作为输出结果
        else:
            new_vel = np.array([ptx_set[line] + temp * dirx_set[line],
                                pty_set[line] + temp * diry_set[line]])

    prog_flag = True
    return prog_flag, new_vel






# if __name__ == '__main__':
#     # 示例
#     states = np.array([137.1168, 2.7222, 1.4377, -0.3481])
#     v_pref = np.array([1.6491, -0.2722])
#     obs_circle = np.array([[160, -25, 20], [160, -77, 20], [160, -123, 20]])
#     obs_poly = [np.array([[40, -40], [80, -40], [80, -95], [40, -95]]),
#                 np.array([[60, 25], [100, 25], [100, 5], [60, 5]]),
#                 np.array([[40, -105], [80, -105], [80, -120], [40, -120]])]
#     obs_dis = 14
#     t_vo = 12
#     v_max = 2.5
#     r_adjust = 5
#
#     v_cmd, flag_collide = static_orca(states, v_pref, obs_circle, obs_poly, obs_dis, t_vo, v_max, r_adjust)
#     print(v_cmd)
#     print(flag_collide)