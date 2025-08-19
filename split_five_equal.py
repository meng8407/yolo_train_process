import os
import cv2
import xml.etree.ElementTree as ET
from xml.dom import minidom
import shutil


def parse_xml(xml_path):
    """解析XML文件，返回图像尺寸和标注框信息"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        objects.append({
            'name': name,
            'xmin': int(bndbox.find('xmin').text),
            'ymin': int(bndbox.find('ymin').text),
            'xmax': int(bndbox.find('xmax').text),
            'ymax': int(bndbox.find('ymax').text)
        })
    return width, height, objects


def create_xml(width, height, objects, output_path, original_filename):
    """创建新的XML文件"""
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = 'images'
    ET.SubElement(annotation, 'filename').text = original_filename

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = '3'

    for obj in objects:
        object_elem = ET.SubElement(annotation, 'object')
        ET.SubElement(object_elem, 'name').text = obj['name']
        bndbox = ET.SubElement(object_elem, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(obj['xmin'])
        ET.SubElement(bndbox, 'ymin').text = str(obj['ymin'])
        ET.SubElement(bndbox, 'xmax').text = str(obj['xmax'])
        ET.SubElement(bndbox, 'ymax').text = str(obj['ymax'])

    rough_string = ET.tostring(annotation, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    with open(output_path, 'w') as f:
        f.write(reparsed.toprettyxml(indent="  "))
    return output_path  # 返回新创建的XML路径


def calculate_regions(img_width, img_height, overlap_ratio=0.1):
    """计算5个分割区域的坐标"""
    center_x, center_y = img_width // 2, img_height // 2
    corner_width = center_x + int(img_width * overlap_ratio / 2)
    corner_height = center_y + int(img_height * overlap_ratio / 2)
    corner_width = min(corner_width, img_width)
    corner_height = min(corner_height, img_height)

    regions = [
        {'name': 'top_left', 'x1': 0, 'y1': 0, 'x2': corner_width, 'y2': corner_height},
        {'name': 'top_right', 'x1': max(0, img_width - corner_width), 'y1': 0, 'x2': img_width, 'y2': corner_height},
        {'name': 'bottom_left', 'x1': 0, 'y1': max(0, img_height - corner_height), 'x2': corner_width,
         'y2': img_height},
        {'name': 'bottom_right', 'x1': max(0, img_width - corner_width), 'y1': max(0, img_height - corner_height),
         'x2': img_width, 'y2': img_height},
        {'name': 'center', 'x1': max(0, center_x - corner_width // 2), 'y1': max(0, center_y - corner_height // 2),
         'x2': min(img_width, center_x + corner_width // 2), 'y2': min(img_height, center_y + corner_height // 2)}
    ]

    # 确保区域尺寸一致
    for region in regions:
        if region['x2'] - region['x1'] != corner_width:
            region['x2'] = region['x1'] + corner_width
        if region['y2'] - region['y1'] != corner_height:
            region['y2'] = region['y1'] + corner_height
    return regions


def filter_and_map_objects(objects, target_labels, label_mapping):
    """根据目标标签列表过滤对象，并进行标签映射"""
    if not target_labels:
        filtered_objects = objects
    else:
        filtered_objects = [obj for obj in objects if obj['name'] in target_labels]

    if label_mapping:
        for obj in filtered_objects:
            if obj['name'] in label_mapping:
                obj['name'] = label_mapping[obj['name']]
    return filtered_objects


def process_objects_in_region(objects, region):
    """处理区域内的目标框"""
    region_objects = []
    for obj in objects:
        x1 = max(obj['xmin'], region['x1'])
        y1 = max(obj['ymin'], region['y1'])
        x2 = min(obj['xmax'], region['x2'])
        y2 = min(obj['ymax'], region['y2'])
        if x1 < x2 and y1 < y2:
            region_objects.append({
                'name': obj['name'],
                'xmin': x1 - region['x1'],
                'ymin': y1 - region['y1'],
                'xmax': x2 - region['x1'],
                'ymax': y2 - region['y1']
            })
    return region_objects

def process_image_and_xml(image_path, xml_path, target_labels, label_mapping, overlap_ratio):
    """处理单个图像和XML文件"""
    # 读取图像并检查有效性
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return None

    try:
        width, height, objects = parse_xml(xml_path)
    except Exception as e:
        print(f"解析XML文件失败 {xml_path}: {e}")
        return None

    # 过滤对象并应用标签映射
    filtered_objects = filter_and_map_objects(objects, target_labels, label_mapping)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    regions = calculate_regions(width, height, overlap_ratio)
    generated_xmls = []  # 存储生成的所有XML路径

    # 获取原始XML所在目录作为输出目录
    output_dir = os.path.dirname(xml_path)  # 关键修改：使用原始XML所在目录

    # 处理每个区域
    for region in regions:
        region_objects = process_objects_in_region(filtered_objects, region)
        if not region_objects and target_labels:
            continue  # 跳过不包含目标标签的区域

        cropped_img = img[region['y1']:region['y2'], region['x1']:region['x2']]
        region_name = region['name']
        output_img_name = f"{base_name}_{region_name}.jpg"
        output_img_path = os.path.join(output_dir, output_img_name)  # 保存到原始目录
        cv2.imwrite(output_img_path, cropped_img)

        output_xml_name = f"{base_name}_{region_name}.xml"
        output_xml_path = os.path.join(output_dir, output_xml_name)  # 保存到原始目录
        cropped_width = region['x2'] - region['x1']
        cropped_height = region['y2'] - region['y1']

        # 创建XML并保存路径
        xml_path_created = create_xml(cropped_width, cropped_height, region_objects, output_xml_path, output_img_name)
        generated_xmls.append(xml_path_created)

    return generated_xmls  # 返回生成的XML路径列表

def main(xml_files, target_labels=None, label_mapping=None, overlap_ratio=0.1):

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    all_xml_paths = []  # 存储所有XML路径（原始+新生成）

    # 添加原始XML文件路径
    all_xml_paths.extend(xml_files)

    for xml_path in xml_files:
        if not os.path.exists(xml_path):
            print(f"警告: XML文件不存在 {xml_path}")
            continue

        # 查找匹配的图像文件
        image_path = None
        base_path = os.path.splitext(xml_path)[0]
        for ext in image_extensions:
            possible_path = base_path + ext
            if os.path.exists(possible_path):
                image_path = possible_path
                break

        if not image_path:
            print(f"警告: 找不到匹配的图像文件 {base_path}[{','.join(image_extensions)}]")
            continue

        print(f"处理: {os.path.basename(image_path)}")
        generated_xmls = process_image_and_xml(
            image_path, xml_path,  # 关键修改：不再传递output_folder
            target_labels, label_mapping, overlap_ratio
        )

        if generated_xmls:
            # 添加新生成的XML路径到总列表
            all_xml_paths.extend(generated_xmls)
            print(f"生成 {len(generated_xmls)} 个新XML文件")

    print(f"处理完成! 共找到 {len(all_xml_paths)} 个XML文件")
    return all_xml_paths  # 返回所有XML路径列表


# 示例用法
if __name__ == "__main__":
    # 1. 输入XML文件列表（绝对路径）
    xml_files = [
        r"C:\Users\developer\Desktop\fzc_ts\image02\8c29e3a04b4911efba26f46b8c9e8c0b_snow.xml",
    ]

    # 2. 目标标签和映射
    target_labels = ['fzc']
    label_mapping = {'fzc': 'fzc'}

    # 3. 执行处理并获取XML路径列表
    all_xml_paths = main(
        xml_files=xml_files,
        target_labels=target_labels,
        label_mapping=label_mapping,
        overlap_ratio=0.1
    )
    print("all_xml_paths", all_xml_paths)

    # 打印所有XML路径
    print("\n所有XML文件路径:")
    for path in all_xml_paths:
        print(path)
