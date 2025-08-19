import os
import sys
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import shutil
import glob
import yaml
from fs_trans import process_xml_list
from split_five_equal import main


def parse_xml(xml_file, target_classes):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # 获取图像尺寸
        size = root.find('size')
        if size is None:
            return [], (None, None)

        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        lines = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in target_classes:
                continue

            # 获取当前类别的索引
            class_idx = target_classes.index(name)

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            # YOLO格式转换
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            line = f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            lines.append(line)

        return lines, (img_width, img_height)
    except Exception as e:
        print(f"Error parsing {xml_file}: {str(e)}")
        return [], (None, None)

def get_image_file(xml_file):
    base_name = os.path.splitext(xml_file)[0]
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    for ext in image_extensions:
        img_file = base_name + ext
        if os.path.exists(img_file):
            return img_file
    return None

def convert_and_save_txt(xml_files, output_dir, dataset_type, target_classes):
    images_output_dir = os.path.join(output_dir, dataset_type, 'images')
    labels_output_dir = os.path.join(output_dir, dataset_type, 'labels')

    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)

    for xml_file in xml_files:
        # 解析XML并获取标注和图像尺寸
        lines, (img_width, img_height) = parse_xml(xml_file, target_classes)
        if img_width is None or img_height is None:
            print(f'Warning: Cannot get image size from {xml_file}')
            continue

        # 获取对应的图像文件
        img_file = get_image_file(xml_file)
        if not img_file:
            print(f'Warning: No corresponding image found for {xml_file}')
            continue

        # 保存图像和标注文件
        img_dest_path = os.path.join(images_output_dir, os.path.basename(img_file))
        txt_dest_path = os.path.join(labels_output_dir, os.path.splitext(os.path.basename(img_file))[0] + '.txt')

        shutil.copy(img_file, img_dest_path)
        with open(txt_dest_path, 'w') as f:
            f.writelines(lines)

def split_dataset(xml_files, output_dir, target_classes, fs_trans=False, split_five_equal=False):
    # 第一次分割：70%训练集，30%临时集
    train_files, temp_files = train_test_split(xml_files, test_size=0.2, random_state=42)

    # 新增处理逻辑（在第一次分割后立即执行）
    if fs_trans:
        train_files = process_xml_list(train_files)  # 应用fs_trans处理
    if split_five_equal:
        train_files = main(train_files, target_labels=target_classes, label_mapping=None, overlap_ratio=0.1)  # 应用split_five_equal处理

    # 第二次分割：临时集分为20%验证集和10%测试集 (2:1比例)
    val_files, test_files = train_test_split(temp_files, test_size=1 / 3, random_state=42)

    # 转换并保存数据集
    convert_and_save_txt(train_files, output_dir, 'train', target_classes)
    convert_and_save_txt(val_files, output_dir, 'val', target_classes)
    convert_and_save_txt(test_files, output_dir, 'test', target_classes)

    return (
        os.path.abspath(os.path.join(output_dir, 'train', 'images')),
        os.path.abspath(os.path.join(output_dir, 'val', 'images')),
        os.path.abspath(os.path.join(output_dir, 'test', 'images')),
        os.path.abspath(os.path.join(output_dir, 'train', 'labels')),
        os.path.abspath(os.path.join(output_dir, 'val', 'labels')),
        os.path.abspath(os.path.join(output_dir, 'test', 'labels'))
    )


def generate_labels_yaml(output_dir, class_names, train_img_path, val_img_path, test_img_path):
    def format_path(path):
        return path.replace('\\', '/')

    labels_yaml_content = {
        'train': format_path(train_img_path),
        'val': format_path(val_img_path),
        'test': format_path(test_img_path),
        'nc': len(class_names),
        'names': class_names
    }

    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(labels_yaml_content, f, default_flow_style=False)

    return yaml_path


def dataset_main(input_dir, output_dir, class_names, fs_trans=False, split_five_equal=False):
    xml_files = glob.glob(os.path.join(input_dir, '*.xml'))

    if not xml_files:
        print('错误: 输入目录中没有找到XML文件。')
        return None

    os.makedirs(output_dir, exist_ok=True)

    # 分割数据集并转换（传递新增参数）
    train_img_path, val_img_path, test_img_path, train_lb_path, val_lb_path, test_lb_path = split_dataset(
        xml_files, output_dir, class_names, fs_trans, split_five_equal)
    yaml_path = generate_labels_yaml(output_dir, class_names, train_img_path, val_img_path, test_img_path)

    # 打印统计信息
    total = len(xml_files)
    train_count = len(glob.glob(os.path.join(output_dir, 'train', 'images', '*')))
    val_count = len(glob.glob(os.path.join(output_dir, 'val', 'images', '*')))
    test_count = len(glob.glob(os.path.join(output_dir, 'test', 'images', '*')))

    print('\n数据转换和分割完成!')
    print(f'总样本数: {total}')
    print(f'训练集样本: {train_count} ({train_count / total:.1%})')
    print(f'验证集样本: {val_count} ({val_count / total:.1%})')
    print(f'测试集样本: {test_count} ({test_count / total:.1%})')
    print(f'目标类别: {", ".join(class_names)} (共{len(class_names)}类)')
    print(f'数据集保存到: {os.path.abspath(output_dir)}')
    print(f'YAML配置文件: {os.path.abspath(yaml_path)}')

    return yaml_path


def convert_dataset(input_dir, output_dir, class_names, fs_trans=False, split_five_equal=False):
    yaml_path = dataset_main(input_dir, output_dir, class_names, fs_trans, split_five_equal)

    if yaml_path:
        print(f'\n成功创建数据集: {os.path.abspath(output_dir)}')
        print(f'YAML文件路径: {os.path.abspath(yaml_path)}')

    return os.path.abspath(yaml_path)

if __name__ == '__main__':
    # 命令行使用方式（支持多类别）
    # 示例: python script.py input_dir output_dir "class1,class2,class3"
    # if len(sys.argv) < 4:
    #     print("使用方法: python script.py <输入目录> <输出目录> <逗号分隔的类别列表>")
    #     print("示例: python script.py ./input ./output \"insulator,bird,car\"")
    #     sys.exit(1)

    # input_dir = sys.argv[1]
    # output_dir = sys.argv[2]
    # class_names = [name.strip() for name in sys.argv[3].split(',')]
    input_dir = r"C:\Users\developer\Desktop\fzc_ts\03"
    output_dir = r'C:\Users\developer\Desktop\fzc_ts\output'
    class_names = ['fzc']

    results = convert_dataset(input_dir, output_dir, class_names,
    fs_trans = True,
    split_five_equal = True   )
    print("results", results)
