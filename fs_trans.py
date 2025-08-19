import cv2
import numpy as np
import os
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom
import sys


def clip_polygon(subject_polygon, clip_box):
    """使用Sutherland-Hodgman算法裁剪多边形"""
    if not subject_polygon:
        return []

    clipped_polygon = list(subject_polygon)
    clip_edges = [
        ('left', clip_box[0]), ('right', clip_box[2]),
        ('top', clip_box[1]), ('bottom', clip_box[3]),
    ]

    for edge, val in clip_edges:
        if not clipped_polygon:
            break

        input_list = clipped_polygon.copy()
        clipped_polygon.clear()

        if not input_list:
            continue

        s = input_list[-1]
        for e in input_list:
            is_s_inside, is_e_inside = False, False

            if edge == 'left':
                is_s_inside, is_e_inside = s[0] >= val, e[0] >= val
            elif edge == 'right':
                is_s_inside, is_e_inside = s[0] <= val, e[0] <= val
            elif edge == 'top':
                is_s_inside, is_e_inside = s[1] >= val, e[1] >= val
            elif edge == 'bottom':
                is_s_inside, is_e_inside = s[1] <= val, e[1] <= val

            if is_e_inside:
                if not is_s_inside:
                    # 计算交点
                    if e[0] != s[0]:
                        slope = (e[1] - s[1]) / (e[0] - s[0])
                    else:
                        slope = float('inf')

                    if edge in ['left', 'right']:
                        intersect_y = s[1] + slope * (val - s[0])
                        intersection = [val, intersect_y]
                    else:
                        intersect_x = s[0] + (val - s[1]) / slope if slope != 0 else s[0]
                        intersection = [intersect_x, val]
                    clipped_polygon.append(intersection)
                clipped_polygon.append(e)
            elif is_s_inside:
                # 计算交点
                if e[0] != s[0]:
                    slope = (e[1] - s[1]) / (e[0] - s[0])
                else:
                    slope = float('inf')

                if edge in ['left', 'right']:
                    intersect_y = s[1] + slope * (val - s[0])
                    intersection = [val, intersect_y]
                else:
                    intersect_x = s[0] + (val - s[1]) / slope if slope != 0 else s[0]
                    intersection = [intersect_x, val]
                clipped_polygon.append(intersection)

            s = e

    return clipped_polygon


def find_image_file(xml_path):
    """根据XML路径查找同名的图像文件"""
    if not os.path.exists(xml_path):
        return None

    base_path = os.path.splitext(xml_path)[0]
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']

    for ext in image_extensions:
        image_path = base_path + ext
        if os.path.exists(image_path):
            return image_path
    return None


def augment_and_update_xml(xml_path: str, stretch_factor: float):
    """
    对单张图片和XML进行透视变换。
    返回：(original_xml_path, transformed_xml_path) 或 None（如果失败）
    """
    try:
        if not os.path.exists(xml_path):
            print(f"  [错误] XML文件不存在: {xml_path}")
            return None

        # 查找对应的图像文件
        image_path = find_image_file(xml_path)
        if image_path is None:
            print(f"  [警告] 找不到与 {xml_path} 对应的图像文件")
            return None

        image = cv2.imread(image_path)
        if image is None:
            print(f"  [警告] 无法读取图片: {image_path}")
            return None

        # 解析XML
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError as e:
            print(f"  [错误] XML解析失败: {xml_path}, 错误: {e}")
            return None

        root = tree.getroot()
        original_height, original_width = image.shape[:2]

        # 定义源点（原始图像的四个角）
        src_points = np.float32([
            [0, 0],
            [original_width - 1, 0],
            [0, original_height - 1],
            [original_width - 1, original_height - 1]
        ])

        # 定义最小变形的比例
        min_stretch_factor_ratio = 0.3

        def get_independent_offset(max_dimension_size, max_stretch_factor):
            min_offset = max_dimension_size * (max_stretch_factor * min_stretch_factor_ratio)
            max_offset = max_dimension_size * max_stretch_factor
            return random.uniform(min_offset, max_offset)

        # 为四个角点生成随机坐标
        p1_new_x = get_independent_offset(original_width, stretch_factor)
        p1_new_y = get_independent_offset(original_height, stretch_factor)

        p2_new_x = original_width - 1 - get_independent_offset(original_width, stretch_factor)
        p2_new_y = get_independent_offset(original_height, stretch_factor)

        p3_new_x = get_independent_offset(original_width, stretch_factor)
        p3_new_y = original_height - 1 - get_independent_offset(original_height, stretch_factor)

        p4_new_x = original_width - 1 - get_independent_offset(original_width, stretch_factor)
        p4_new_y = original_height - 1 - get_independent_offset(original_height, stretch_factor)

        dst_points = np.float32([
            [p1_new_x, p1_new_y],
            [p2_new_x, p2_new_y],
            [p3_new_x, p3_new_y],
            [p4_new_x, p4_new_y]
        ])

        # 计算变换矩阵并应用到图像
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        # 对原始图像进行透视变换，使用128灰度作为边界填充
        warped_image = cv2.warpPerspective(
            image, transform_matrix, (original_width, original_height),
            borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128)
        )

        # 更新XML中的边界框坐标
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue

            try:
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))
            except (AttributeError, ValueError) as e:
                print(f"  [警告] 无效的bbox坐标: {xml_path}, 错误: {e}")
                continue

            original_box_corners = np.float32([
                [[xmin, ymin]],
                [[xmax, ymin]],
                [[xmax, ymax]],
                [[xmin, ymax]]
            ])

            try:
                transformed_quad = cv2.perspectiveTransform(
                    original_box_corners, transform_matrix
                ).reshape(4, 2).tolist()
            except cv2.error as e:
                print(f"  [错误] 坐标变换失败: {xml_path}, 错误: {e}")
                continue

            image_boundary = (0, 0, original_width - 1, original_height - 1)
            clipped_visible_polygon = clip_polygon(transformed_quad, image_boundary)

            if not clipped_visible_polygon:
                root.remove(obj)
                continue

            clipped_poly_np = np.array(clipped_visible_polygon)
            new_xmin, new_ymin = np.min(clipped_poly_np, axis=0)
            new_xmax, new_ymax = np.max(clipped_poly_np, axis=0)

            # 确保坐标在图像范围内
            new_xmin = max(0, int(new_xmin))
            new_ymin = max(0, int(new_ymin))
            new_xmax = min(original_width - 1, int(new_xmax))
            new_ymax = min(original_height - 1, int(new_ymax))

            if new_xmin >= new_xmax or new_ymin >= new_ymax:
                root.remove(obj)
                continue

            bndbox.find('xmin').text = str(new_xmin)
            bndbox.find('ymin').text = str(new_ymin)
            bndbox.find('xmax').text = str(new_xmax)
            bndbox.find('ymax').text = str(new_ymax)

        # 保存变换后的图像
        base_path = os.path.splitext(xml_path)[0]
        image_ext = os.path.splitext(image_path)[1]
        new_image_path = f"{base_path}_warp{image_ext}"
        cv2.imwrite(new_image_path, warped_image)

        # 保存变换后的XML
        new_xml_path = f"{base_path}_warp.xml"

        # 确保XML格式正确
        try:
            xml_string = ET.tostring(root, encoding='utf-8')
            pretty_xml = minidom.parseString(xml_string).toprettyxml(indent="  ")
            with open(new_xml_path, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)
        except Exception as e:
            print(f"  [错误] 保存XML失败: {new_xml_path}, 错误: {e}")
            return None

        return (xml_path, new_xml_path)

    except Exception as e:
        print(f"  [错误] 处理文件 {xml_path} 时发生错误: {str(e)}")
        return None


def process_xml_list(xml_paths: list, stretch_factor: float = 0.22, p: float = 1.0):
    """
    处理XML文件列表，对每个文件进行仿射变换
    参数:
        xml_paths: XML文件路径列表
        stretch_factor: 拉伸变形因子 (默认0.22)
        p: 处理概率 (默认1.0，即处理所有文件)
    返回:
        list: 包含所有XML路径的列表（原始和变换后的）
    """
    if not isinstance(xml_paths, list):
        print("[错误] 输入参数xml_paths必须是一个列表")
        return []

    all_xml_paths = []  # 包含原始和变换后的所有XML路径

    total_to_process = len(xml_paths)
    for i, xml_path in enumerate(xml_paths):
        print(f"[{i + 1}/{total_to_process}] 处理: {xml_path}")

        if not isinstance(xml_path, str) or not xml_path.endswith('.xml'):
            print(f"  [警告] 跳过无效的XML路径: {xml_path}")
            continue

        # 总是保留原始XML路径
        if xml_path not in all_xml_paths:
            all_xml_paths.append(xml_path)

        # 根据概率决定是否处理
        if random.random() > p:
            continue

        result = augment_and_update_xml(xml_path, stretch_factor)
        if result is not None:
            original_xml, transformed_xml = result
            if transformed_xml not in all_xml_paths:
                all_xml_paths.append(transformed_xml)

    return all_xml_paths


if __name__ == "__main__":
    # 示例用法
    xml_paths = [
        r"D:\Yujing\PycharmProjects\detection\detection_yolo\detection_v8\fs_test\2.xml",
        r"D:\Yujing\PycharmProjects\detection\detection_yolo\detection_v8\fs_test\2_warp.xml",
    ]

    # 检查测试目录是否存在
    # if not os.path.exists("./test_data"):
    #     os.makedirs("./test_data")
    #     print("已创建测试目录，请在其中放置测试文件")
    #     sys.exit(0)

    stretch_factor = 0.22  # 控制拉伸变形程度的因子
    p = 1.0  # 处理概率

    results = process_xml_list(xml_paths, stretch_factor, p)
    print(results)

    print("\n处理完成，结果XML路径列表:")
    for xml_path in results:
        print(xml_path)