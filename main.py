import taichi as ti
import random

# 初始化 Taichi GPU 后端 (Mac 自动调用 Metal，Win 调用 CUDA/Vulkan)
ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 交互参数
light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())
samples_per_pixel = ti.field(ti.i32, shape=())  # MSAA采样数

# 材质常量枚举
MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2  # 新增玻璃材质


@ti.func
def normalize(v):
    return v / v.norm(1e-5)


@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N


@ti.func
def refract(I, N, ior):
    """
    基于斯涅尔定律计算折射光线
    I: 入射光线方向（指向表面）
    N: 表面法线（指向外部）
    ior: 折射率（空气/玻璃）
    返回: (是否发生折射, 折射光线方向)
    """
    cos_theta = ti.min(I.dot(-N), 1.0)
    sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)

    # 检查是否发生全反射
    if ior * sin_theta > 1.0:
        return False, ti.Vector([0.0, 0.0, 0.0])

    # 计算折射光线
    r_out_perp = ior * (I + cos_theta * N)
    r_out_parallel = -ti.sqrt(ti.abs(1.0 - r_out_perp.norm_sqr())) * N
    return True, normalize(r_out_perp + r_out_parallel)


@ti.func
def intersect_sphere(ro, rd, center, radius):
    """球体求交，返回 (距离 t, 法线 normal)"""
    t = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc = ro - center
    b = 2.0 * oc.dot(rd)
    c = oc.dot(oc) - radius * radius
    delta = b * b - 4.0 * c
    if delta > 0:
        t1 = (-b - ti.sqrt(delta)) / 2.0
        if t1 > 0:
            t = t1
            p = ro + rd * t
            normal = normalize(p - center)
    return t, normal


@ti.func
def intersect_plane(ro, rd, plane_y):
    """水平无限大平面求交"""
    t = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])  # 法线永远朝上
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 0:
            t = t1
    return t, normal


@ti.func
def scene_intersect(ro, rd):
    """
    遍历场景，寻找最近交点。
    返回: (t, 法线 N, 颜色 color, 材质 mat_id)
    """
    min_t = 1e10
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    # 1. 检测玻璃球（原红色球）
    t, n = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.9, 0.95, 1.0])  # 淡蓝色玻璃
        hit_mat = MAT_GLASS

    # 2. 检测银色镜面球
    t, n = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.9, 0.9, 0.9])  # 镜面反射基础色
        hit_mat = MAT_MIRROR

    # 3. 检测地板
    t, n = intersect_plane(ro, rd, -1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_mat = MAT_DIFFUSE
        # 生成棋盘格纹理
        p = ro + rd * t
        grid_scale = 2.0
        ix = ti.floor(p.x * grid_scale)
        iz = ti.floor(p.z * grid_scale)
        # 判断坐标的奇偶性来交替颜色
        if (ix + iz) % 2 == 0:
            hit_c = ti.Vector([0.3, 0.3, 0.3])  # 灰色格子
        else:
            hit_c = ti.Vector([0.8, 0.8, 0.8])  # 白色格子

    return min_t, hit_n, hit_c, hit_mat


@ti.kernel
def render():
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    bg_color = ti.Vector([0.05, 0.15, 0.2])
    ior = 1.5  # 玻璃折射率

    for i, j in pixels:
        pixel_color = ti.Vector([0.0, 0.0, 0.0])

        # MSAA多重采样
        for s in range(samples_per_pixel[None]):
            # 在像素内添加随机偏移
            offset_x = ti.random() - 0.5
            offset_y = ti.random() - 0.5

            u = ((i + offset_x) - res_x / 2.0) / res_y * 2.0
            v = ((j + offset_y) - res_y / 2.0) / res_y * 2.0

            ro = ti.Vector([0.0, 1.0, 5.0])  # 摄像机稍微抬高一点
            rd = normalize(ti.Vector([u, v - 0.2, -1.0]))  # 视角微微向下看

            final_color = ti.Vector([0.0, 0.0, 0.0])
            throughput = ti.Vector([1.0, 1.0, 1.0])  # 光线能量吞吐量

            # 迭代式光线追踪（代替递归）
            for bounce in range(max_bounces[None]):
                t, N, obj_color, mat_id = scene_intersect(ro, rd)

                # 如果没击中任何物体，加上背景色并结束追踪
                if t > 1e9:
                    final_color += throughput * bg_color
                    break

                p = ro + rd * t

                # 分支 1：镜面反射材质
                if mat_id == MAT_MIRROR:
                    # 生成反射射线，注意必须要加上极其微小的法线偏移（1e-4）防止自相交！
                    ro = p + N * 1e-4
                    rd = normalize(reflect(rd, N))
                    # 镜面吸收一部分能量 (反射率 0.8)
                    throughput *= 0.8 * obj_color
                    # 不跳出循环，继续追踪反射射线

                # 分支 2：漫反射材质
                elif mat_id == MAT_DIFFUSE:
                    L = normalize(light_pos - p)

                    # --- 硬阴影检测 ---
                    # 从当前交点向光源发射暗影射线，同样需要法线偏移
                    shadow_ray_orig = p + N * 1e-4
                    shadow_t, _, _, _ = scene_intersect(shadow_ray_orig, L)

                    # 判断：如果去光源的路上没被挡住 (或者遮挡物比光源还远)，则计算光照
                    dist_to_light = (light_pos - p).norm()
                    in_shadow = 0.0
                    if shadow_t < dist_to_light:
                        in_shadow = 1.0  # 被挡住了！

                    # 简单的 Phong 光照 (由于是 Whitted-style，只算直接光)
                    ambient = 0.2 * obj_color

                    # 【已修复】：在 if 外部提前声明并初始化 direct_light
                    direct_light = ambient

                    # 如果不在阴影里，再额外加上漫反射的光
                    if in_shadow == 0.0:
                        diff = ti.max(0.0, N.dot(L))
                        diffuse = 0.8 * diff * obj_color
                        direct_light += diffuse

                    # 将当前点的颜色乘以积累的能量，加到最终颜色里
                    final_color += throughput * direct_light

                    # 漫反射表面会打散光线，Whitted 风格下主射线到此终止
                    break

                # 分支 3：玻璃材质
                elif mat_id == MAT_GLASS:
                    # 计算反射和折射
                    refracted, refracted_dir = refract(rd, N, ior)

                    if refracted:
                        # 发生折射，光线进入玻璃内部
                        ro = p - N * 1e-4  # 向内部偏移防止自相交
                        rd = refracted_dir
                        throughput *= obj_color  # 玻璃吸收少量能量
                    else:
                        # 发生全反射
                        ro = p + N * 1e-4
                        rd = normalize(reflect(rd, N))
                        throughput *= obj_color

        # 平均所有采样的颜色
        pixels[i, j] = ti.math.clamp(pixel_color / samples_per_pixel[None], 0.0, 1.0)


def main():
    window = ti.ui.Window("Ray Tracing Demo", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    # 初始化参数
    light_pos_x[None] = 2.0
    light_pos_y[None] = 4.0
    light_pos_z[None] = 3.0
    max_bounces[None] = 5  # 玻璃需要更多弹射次数
    samples_per_pixel[None] = 4  # 默认4倍MSAA

    while window.running:
        render()
        canvas.set_image(pixels)

        with gui.sub_window("Controls", 0.75, 0.05, 0.23, 0.3):
            light_pos_x[None] = gui.slider_float('Light X', light_pos_x[None], -5.0, 5.0)
            light_pos_y[None] = gui.slider_float('Light Y', light_pos_y[None], 1.0, 8.0)
            light_pos_z[None] = gui.slider_float('Light Z', light_pos_z[None], -5.0, 5.0)
            max_bounces[None] = gui.slider_int('Max Bounces', max_bounces[None], 1, 10)
            samples_per_pixel[None] = gui.slider_int('MSAA Samples', samples_per_pixel[None], 1, 16)

        window.show()


if __name__ == '__main__':
    main()