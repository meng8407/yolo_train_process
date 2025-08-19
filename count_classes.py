# -*- coding:utf-8 -*-
import os
import xml.etree.ElementTree as ET

def count_classes(xml_dir, predefined_class_file=None):

    classCount = {}

    if predefined_class_file and os.path.exists(predefined_class_file):
        with open(predefined_class_file, 'r', encoding='utf-8') as f:
            for line in f:
                c = line.strip()
                if c:
                    classCount[c] = 0

    def parse_file(path):
        try:
            root = ET.parse(path).getroot()
            for obj in root.iter('object'):
                name = obj.find('name').text
                if predefined_class_file:
                    if name in classCount:
                        classCount[name] += 1
                else:
                    classCount[name] = classCount.get(name, 0) + 1
        except Exception as e:
            print(f"Error parsing {path}: {str(e)}")

    for root, dirs, files in os.walk(xml_dir):
        for file in files:
            if file.endswith(".xml"):
                xml_path = os.path.join(root, file)
                parse_file(xml_path)

    return classCount


def get_class_name(xml_dir):
    results = count_classes(xml_dir)
    print("results",results)
    print("\n类别统计结果:")
    for cls, count in results.items():
        print(f'{cls} : {count}')
    print(f"\n总计类别数: {len(results)}")
    # return


if __name__ == '__main__':
    xml_dir=r"C:\Users\developer\Desktop\fzc_ts\image02"
    get_class_name(xml_dir)