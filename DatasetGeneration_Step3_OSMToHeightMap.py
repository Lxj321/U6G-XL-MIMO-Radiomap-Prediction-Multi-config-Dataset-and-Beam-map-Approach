import xml.dom.minidom
import math
import os
import glob
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.errors import InvalidGeometryError

# -------------------------------
# 配置参数（可根据需求调整）
# -------------------------------
AREA_SIZE = (1280, 1280)  # 区域大小 (米)，宽×高
RESOLUTION = 5  # 高度矩阵分辨率（米/栅格）
HEIGHT_PER_LEVEL = 5  # 每层楼高度（米），用于计算建筑高度


# -------------------------------
# 1. 解析OSM文件，提取建筑物数据
# -------------------------------
def parse_osm(osm_file_path):
    """从OSM文件中提取建筑物轮廓和高度信息"""
    try:
        doc = xml.dom.minidom.parse(osm_file_path)
    except Exception as e:
        print(f"读取OSM文件失败 {osm_file_path}: {e}")
        return []

    # 提取所有节点（经纬度坐标）
    nodes = doc.getElementsByTagName("node")
    id_to_coord = {}  # {node_id: (lon, lat)}
    for node in nodes:
        if node.hasAttribute('id') and node.hasAttribute('lon') and node.hasAttribute('lat'):
            node_id = node.attributes['id'].value
            lon = float(node.attributes['lon'].value)
            lat = float(node.attributes['lat'].value)
            id_to_coord[node_id] = (lon, lat)

    # 提取所有建筑物
    buildings = []  # 每个元素: (轮廓坐标列表, 高度)
    ways = doc.getElementsByTagName("way")
    for way in ways:
        # 判断是否为建筑物
        is_building = False
        building_levels = 1  # 默认1层
        tags = way.getElementsByTagName('tag')
        for tag in tags:
            if tag.hasAttribute('k') and tag.hasAttribute('v'):
                if tag.attributes['k'].value == 'building':
                    is_building = True
                if tag.attributes['k'].value == 'building:levels':
                    try:
                        building_levels = int(tag.attributes['v'].value)
                    except ValueError:
                        building_levels = 1  # 无效值默认1层

        if not is_building:
            continue

        # 提取建筑物轮廓节点
        nds = way.getElementsByTagName('nd')
        node_ids = [nd.attributes['ref'].value for nd in nds if nd.hasAttribute('ref')]
        if len(node_ids) < 4:  # 至少4个点才能构成闭合轮廓
            continue

        # 获取经纬度坐标
        coords = []
        for node_id in node_ids:
            if node_id in id_to_coord:
                coords.append(id_to_coord[node_id])
        if len(coords) <4:
            continue

        # 计算建筑高度（楼层数×层高）
        if building_levels==1:
            building_height = 20
        else:
            if building_levels>1:
                building_height = building_levels * HEIGHT_PER_LEVEL
            else:
                building_levels=0
        buildings.append((coords, building_height))

    return buildings


# -------------------------------
# 2. 坐标转换（经纬度→平面坐标）
# -------------------------------
def convert_coords(buildings):
    """将经纬度坐标转换为相对于区域中心的平面坐标（米）"""
    if not buildings:
        return []

    # 收集所有坐标计算中心
    all_lons = []
    all_lats = []
    for (coords, _) in buildings:
        for (lon, lat) in coords:
            all_lons.append(lon)
            all_lats.append(lat)
    center_lon = sum(all_lons) / len(all_lons) if all_lons else 0
    center_lat = sum(all_lats) / len(all_lats) if all_lats else 0

    # 经纬度→米坐标转换（简化的墨卡托转换）
    def lonlat_to_xy(lon, lat):
        # 地球半径（米）
        R = 6378137.0
        # 转换为弧度
        lon_rad = math.radians(lon)
        lat_rad = math.radians(lat)
        # 墨卡托投影
        x = R * lon_rad
        y = R * math.log(math.tan(math.pi/4 + lat_rad/2))
        return (x, y)

    # 转换所有建筑物坐标并居中
    center_x, center_y = lonlat_to_xy(center_lon, center_lat)
    converted_buildings = []
    for (coords, height) in buildings:
        xy_coords = []
        for (lon, lat) in coords:
            x, y = lonlat_to_xy(lon, lat)
            # 相对于中心偏移，转换为区域内坐标
            x_rel = x - center_x
            y_rel = y - center_y
            xy_coords.append((x_rel, y_rel))
        converted_buildings.append((xy_coords, height))

    return converted_buildings


# -------------------------------
# 3. 裁剪超出区域的建筑物
# -------------------------------
def crop_buildings(buildings):
    """保留区域范围内的建筑物（基于AREA_SIZE）"""
    cropped = []
    half_w, half_h = AREA_SIZE[0]/2, AREA_SIZE[1]/2
    for (coords, height) in buildings:
        # 判断建筑物是否有部分在区域内
        in_bounds = False
        for (x, y) in coords:
            if -half_w <= x <= half_w and -half_h <= y <= half_h:
                in_bounds = True
                break
        if in_bounds:
            cropped.append((coords, height))
    return cropped







# -------------------------------
# 4. 生成高度矩阵
# -------------------------------
def generate_height_matrix(buildings):
    """生成2D高度矩阵，确保建筑物内部填充高度"""
    # 计算矩阵尺寸
    half_w, half_h = AREA_SIZE[0]/2, AREA_SIZE[1]/2
    cols = int(AREA_SIZE[0] / RESOLUTION)  # 宽度方向栅格数
    rows = int(AREA_SIZE[1] / RESOLUTION)  # 高度方向栅格数

    # 初始化高度矩阵（0表示无建筑物）
    height_matrix = np.zeros((rows, cols), dtype=np.float32)

    # 生成栅格中心坐标
    x_centers = np.linspace(-half_w + RESOLUTION/2, half_w - RESOLUTION/2, cols)
    y_centers = np.linspace(-half_h + RESOLUTION/2, half_h - RESOLUTION/2, rows)

    # 处理每个建筑物
    for idx, (coords, height) in enumerate(buildings):
        try:
            # 创建建筑物轮廓多边形（闭合）
            if coords[0] != coords[-1]:
                coords = coords + [coords[0]]  # 闭合多边形
            polygon = Polygon(coords)

            # 过滤无效多边形
            if not polygon.is_valid:
                polygon = polygon.buffer(0)  # 修复无效多边形
            if not polygon.is_valid:
                continue

            # 填充建筑物内部栅格高度
            for i, x in enumerate(x_centers):
                for j, y in enumerate(y_centers):
                    # 判断点是否在建筑物内部
                    point = Point(x, y)
                    if polygon.contains(point) or polygon.touches(point):
                        # 若有重叠建筑，取最高高度
                        if height > height_matrix[j, i]:
                            height_matrix[j, i] = height

            print(f"处理建筑物 {idx+1}/{len(buildings)}")

        except InvalidGeometryError:
            print(f"建筑物 {idx+1} 轮廓无效，已跳过")
        except Exception as e:
            print(f"处理建筑物 {idx+1} 出错: {e}")

    return height_matrix, (x_centers, y_centers)


# -------------------------------
# 5. 保存高度矩阵
# -------------------------------
def save_matrix(height_matrix, coords, output_path):
    """保存高度矩阵和坐标信息"""
    # 保存高度矩阵
    np.save(output_path, height_matrix)
    # 保存坐标范围（用于后续定位）
    x_centers, y_centers = coords
    np.savez(
        output_path.replace(".npy", "_coords.npz"),
        min_x=x_centers[0],
        max_x=x_centers[-1],
        min_y=y_centers[0],
        max_y=y_centers[-1],
        resolution=RESOLUTION
    )
    print(f"高度矩阵已保存至: {output_path}")
    print(f"矩阵尺寸: {height_matrix.shape} (行×列)")
    print(f"区域范围: X [{x_centers[0]:.1f}, {x_centers[-1]:.1f}] 米, Y [{y_centers[0]:.1f}, {y_centers[-1]:.1f}] 米")


# -------------------------------
# 处理单个文件的流程
# -------------------------------
def process_file(input_file, output_file):
    print(f"\n===== 处理文件: {os.path.basename(input_file)} =====")
    # 1. 解析OSM获取建筑物数据
    buildings = parse_osm(input_file)
    if not buildings:
        print("未提取到建筑物数据")
        return
    print(f"提取到 {len(buildings)} 个建筑物")

    # 2. 坐标转换（经纬度→平面坐标）
    converted_buildings = convert_coords(buildings)

    # 3. 裁剪区域外建筑物
    cropped_buildings = crop_buildings(converted_buildings)
    print(f"区域内保留 {len(cropped_buildings)} 个建筑物")

    # 4. 生成高度矩阵
    height_matrix, coords = generate_height_matrix(cropped_buildings)

    # 5. 保存结果
    save_matrix(height_matrix, coords, output_file)


# -------------------------------
# 主流程：批量处理OSM文件
# -------------------------------
def main():
    input_dir = "/Users/xiaojieli/Downloads/SEU_Research_XLMIMO_RadioMap/Dataset_Test/dataset"  # 输入OSM文件夹
    output_base_dir = "/Users/xiaojieli/Downloads/SEU_Research_XLMIMO_RadioMap/Dataset_Test/heightmaps"  # 输出矩阵文件夹
    os.makedirs(output_base_dir, exist_ok=True)

    # 获取所有OSM文件并排序
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.osm")))
    if not input_files:
        print("未找到任何OSM文件！")
        return

    # 批量处理
    for idx, input_file in enumerate(input_files, start=1):
        output_folder = os.path.join(output_base_dir, f"u{idx}")
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f"u{idx}_height_matrix.npy")
        process_file(input_file, output_file)

    print("\n===== 所有文件处理完成 =====")


if __name__ == "__main__":
    main()