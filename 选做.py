import taichi as ti

# 初始化 Taichi GPU 后端 (Mac 自动调用 Metal，Win 调用 CUDA/Vulkan)
ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 交互参数
light_pos_x = ti.field(ti.f32, shape=())
light_pos_y = ti.field(ti.f32, shape=())
light_pos_z = ti.field(ti.f32, shape=())
max_bounces = ti.field(ti.i32, shape=())

# 新增：抗锯齿采样次数
samples_per_pixel = ti.field(ti.i32, shape=())

# 材质常量枚举
MAT_DIFFUSE = 0
MAT_MIRROR = 1
MAT_GLASS = 2

# 玻璃折射率常量
IOR_AIR = 1.0
IOR_GLASS = 1.5


@ti.func
def normalize(v):
    return v / v.norm(1e-5)


@ti.func
def reflect(I, N):
    return I - 2.0 * I.dot(N) * N


@ti.func
def refract(I, N, ior):
    """根据斯涅尔定律计算折射方向"""
    cos_i = -ti.max(-1.0, ti.min(1.0, I.dot(N)))
    n = N
    eta = ior
    if cos_i < 0:
        cos_i = -cos_i
        n = -N
        eta = 1.0 / ior
    k = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    total_internal_reflection = False
    refracted_dir = ti.Vector([0.0, 0.0, 0.0])
    if k < 0.0:
        total_internal_reflection = True
    else:
        refracted_dir = normalize(eta * I + (eta * cos_i - ti.sqrt(k)) * n)
    return refracted_dir, total_internal_reflection


@ti.func
def fresnel_schlick(cos_i, ior):
    """Schlick 近似菲涅尔方程"""
    r0 = (1.0 - ior) / (1.0 + ior)
    r0 = r0 * r0
    return r0 + (1.0 - r0) * ti.pow(1.0 - cos_i, 5.0)


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
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 0:
            t = t1
    return t, normal


@ti.func
def scene_intersect(ro, rd):
    """遍历场景，寻找最近交点。返回: (t, 法线 N, 颜色 color, 材质 mat_id)"""
    min_t = 1e10
    hit_n = ti.Vector([0.0, 0.0, 0.0])
    hit_c = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    # 1. 检测玻璃球
    t, n = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.95, 0.95, 1.0])
        hit_mat = MAT_GLASS

    # 2. 检测银色镜面球
    t, n = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_c = ti.Vector([0.9, 0.9, 0.9])
        hit_mat = MAT_MIRROR

    # 3. 检测地板
    t, n = intersect_plane(ro, rd, -1.0)
    if 0 < t < min_t:
        min_t = t
        hit_n = n
        hit_mat = MAT_DIFFUSE
        p = ro + rd * t
        grid_scale = 2.0
        ix = ti.floor(p.x * grid_scale)
        iz = ti.floor(p.z * grid_scale)
        if (ix + iz) % 2 == 0:
            hit_c = ti.Vector([0.3, 0.3, 0.3])
        else:
            hit_c = ti.Vector([0.8, 0.8, 0.8])

    return min_t, hit_n, hit_c, hit_mat


@ti.func
def trace_ray(ro, rd, light_pos, bg_color, max_bounce):
    """追踪单条光线，返回最终颜色"""
    final_color = ti.Vector([0.0, 0.0, 0.0])
    throughput = ti.Vector([1.0, 1.0, 1.0])

    for bounce in range(max_bounce):
        t, N, obj_color, mat_id = scene_intersect(ro, rd)

        if t > 1e9:
            final_color += throughput * bg_color
            break

        p = ro + rd * t

        if mat_id == MAT_MIRROR:
            ro = p + N * 1e-4
            rd = normalize(reflect(rd, N))
            throughput *= 0.8 * obj_color

        elif mat_id == MAT_GLASS:
            entering = rd.dot(N) < 0.0
            eta = 1.0
            cos_i = 0.0
            normal = ti.Vector([0.0, 0.0, 0.0])
            if entering:
                eta = IOR_AIR / IOR_GLASS
                cos_i = -rd.dot(N)
                normal = N
            else:
                eta = IOR_GLASS / IOR_AIR
                cos_i = rd.dot(-N)
                normal = -N

            refracted_dir, tir = refract(rd, normal, eta)
            fresnel_R = fresnel_schlick(cos_i, eta)

            if tir:
                ro = p + normal * 1e-4
                rd = normalize(reflect(rd, normal))
                throughput *= 0.95 * obj_color
            else:
                if fresnel_R > 0.5:
                    ro = p + normal * 1e-4
                    rd = normalize(reflect(rd, normal))
                    throughput *= fresnel_R * 0.95 * obj_color
                else:
                    ro = p - normal * 1e-4
                    rd = refracted_dir
                    throughput *= (1.0 - fresnel_R) * 0.95 * obj_color

        elif mat_id == MAT_DIFFUSE:
            L = normalize(light_pos - p)
            shadow_ray_orig = p + N * 1e-4
            shadow_t, _, _, _ = scene_intersect(shadow_ray_orig, L)
            dist_to_light = (light_pos - p).norm()
            in_shadow = 0.0
            if shadow_t < dist_to_light:
                in_shadow = 1.0
            ambient = 0.2 * obj_color
            direct_light = ambient
            if in_shadow == 0.0:
                diff = ti.max(0.0, N.dot(L))
                diffuse = 0.8 * diff * obj_color
                direct_light += diffuse
            final_color += throughput * direct_light
            break

    return final_color


@ti.kernel
def render():
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    bg_color = ti.Vector([0.05, 0.15, 0.2])
    spp = samples_per_pixel[None]

    for i, j in pixels:
        accumulated_color = ti.Vector([0.0, 0.0, 0.0])

        for s in range(spp):
            seed = i * 73856093 + j * 19349663 + s * 83492791
            rand_x = ti.cast(seed % 1000, ti.f32) / 1000.0
            seed = seed * 1103515245 + 12345
            rand_y = ti.cast(seed % 1000, ti.f32) / 1000.0

            offset_x = (rand_x - 0.5) / res_y * 2.0
            offset_y = (rand_y - 0.5) / res_y * 2.0

            u = (i - res_x / 2.0) / res_y * 2.0 + offset_x
            v = (j - res_y / 2.0) / res_y * 2.0 + offset_y

            ro = ti.Vector([0.0, 1.0, 5.0])
            rd = normalize(ti.Vector([u, v - 0.2, -1.0]))

            ray_color = trace_ray(ro, rd, light_pos, bg_color, max_bounces[None])
            accumulated_color += ray_color

        final_color = accumulated_color / ti.cast(spp, ti.f32)
        pixels[i, j] = ti.math.clamp(final_color, 0.0, 1.0)


def main():
    window = ti.ui.Window("Ray Tracing Demo - Glass + MSAA", (res_x, res_y))
    canvas = window.get_canvas()
    gui = window.get_gui()

    light_pos_x[None] = 2.0
    light_pos_y[None] = 4.0
    light_pos_z[None] = 3.0
    max_bounces[None] = 5
    samples_per_pixel[None] = 4

    while window.running:
        render()
        canvas.set_image(pixels)

        with gui.sub_window("Controls", 0.72, 0.05, 0.26, 0.28):
            light_pos_x[None] = gui.slider_float('Light X', light_pos_x[None], -5.0, 5.0)
            light_pos_y[None] = gui.slider_float('Light Y', light_pos_y[None], 1.0, 8.0)
            light_pos_z[None] = gui.slider_float('Light Z', light_pos_z[None], -5.0, 5.0)
            max_bounces[None] = gui.slider_int('Max Bounces', max_bounces[None], 1, 10)
            samples_per_pixel[None] = gui.slider_int('MSAA Samples', samples_per_pixel[None], 1, 16)

        window.show()


if __name__ == '__main__':
    main()