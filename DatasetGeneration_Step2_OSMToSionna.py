import bpy
import bmesh
import xml.dom.minidom
import math
import mathutils
import os
import glob
from shapely.geometry import Polygon
from shapely.errors import InvalidGeometryError


# -------------------------------
# 1. 清空场景
# -------------------------------
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for scene in list(bpy.data.scenes):
        if scene != bpy.context.scene:
            bpy.data.scenes.remove(scene)
    bpy.ops.outliner.orphans_purge(do_recursive=True)


# -------------------------------
# 2. 导入 OSM 文件并生成建筑
# -------------------------------
def import_osm(osm_file_path):
    def get_or_create_material(name, diffuse_color):
        if name in bpy.data.materials:
            mat = bpy.data.materials[name]
        else:
            mat = bpy.data.materials.new(name)
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            for node in nodes:
                nodes.remove(node)
            node_out = nodes.new(type='ShaderNodeOutputMaterial')
            node_diff = nodes.new(type='ShaderNodeBsdfDiffuse')
            node_diff.inputs['Color'].default_value = diffuse_color
            mat.node_tree.links.new(node_diff.outputs[0], node_out.inputs[0])
        return mat

    # 全局材质定义
    mat_wall_global = get_or_create_material("itu_marble", (1, 0.5, 0.2, 1))
    mat_roof_global = get_or_create_material("itu_metal", (0.29, 0.25, 0.21, 1))
    mat_concrete_global = get_or_create_material("itu_concrete", (0.5, 0.5, 0.5, 1))

    try:
        doc = xml.dom.minidom.parse(osm_file_path)
    except Exception as e:
        print(f"Error reading OSM file {osm_file_path}: {e}")
        return

    # 提取所有节点（经纬度坐标）
    nodes = doc.getElementsByTagName("node")
    id_to_tuple = {}
    for node in nodes:
        if node.hasAttribute('id') and node.hasAttribute('lon') and node.hasAttribute('lat'):
            node_id = node.attributes['id'].value
            lon = float(node.attributes['lon'].value)
            lat = float(node.attributes['lat'].value)
            id_to_tuple[node_id] = (lon, lat)

    # 提取所有建筑物
    all_ways = doc.getElementsByTagName("way")
    buildings = []
    for way in all_ways:
        is_building = False
        tags = way.getElementsByTagName('tag')
        for tag in tags:
            if tag.hasAttribute('k') and tag.hasAttribute('v') and tag.attributes['k'].value == 'building':
                is_building = True
                break
        if is_building:
            buildings.append(way)

    all_buildings = []
    for b in buildings:
        nds = b.getElementsByTagName('nd')
        node_ids = [nd.attributes['ref'].value for nd in nds if nd.hasAttribute('ref')]
        
        lst = []
        for node_id in node_ids:
            if node_id in id_to_tuple:
                lst.append(id_to_tuple[node_id])
        
        if len(lst) < 4:
            print(f"跳过顶点不足4个的建筑（{len(lst)}个顶点）")
            continue
            
        tags = b.getElementsByTagName('tag')
        level = 1
        for tag in tags:
            if tag.tagName == 'tag' and tag.attributes['k'].value == 'building:levels':
                try:
                    level = int(tag.attributes['v'].value)
                except ValueError:
                    level = 1
        height = 20 if level == 1 else level * 5
        all_buildings.append((lst, height))

    # 计算中心经纬度
    all_lons = []
    all_lats = []
    for (coords, _) in all_buildings:
        for (lon, lat) in coords:
            all_lons.append(lon)
            all_lats.append(lat)
    center_lon = sum(all_lons) / len(all_lons) if all_lons else 0
    center_lat = sum(all_lats) / len(all_lats) if all_lats else 0

    # 墨卡托投影转换
    def lonlat_to_xy(lon, lat):
        R = 6378137.0
        lon_rad = math.radians(float(lon))
        lat_rad = math.radians(float(lat))
        x = R * lon_rad
        y = R * math.log(math.tan(math.pi/4 + lat_rad/2))
        return (x, y)

    center_x, center_y = lonlat_to_xy(center_lon, center_lat)

    # 转换建筑物坐标并处理轮廓
    buildings_xy = []
    for lst, height in all_buildings:
        tmp = []
        for coord in lst:
            x, y = lonlat_to_xy(coord[0], coord[1])
            x_rel = x - center_x
            y_rel = y - center_y
            tmp.append((x_rel, y_rel))
        
        try:
            if tmp[0] != tmp[-1]:
                tmp.append(tmp[0])
            
            polygon = Polygon(tmp)
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
                if not polygon.is_valid:
                    print("修复无效轮廓失败，跳过该建筑")
                    continue
            
            if polygon.geom_type == 'Polygon':
                repaired_coords = list(polygon.exterior.coords)[:-1]
                buildings_xy.append((repaired_coords, height))
            else:
                print("轮廓修复后不是简单多边形，跳过该建筑")
        
        except InvalidGeometryError:
            print("轮廓无效，跳过该建筑")
        except Exception as e:
            print(f"处理轮廓时出错: {e}，跳过该建筑")

    # 创建建筑物
    scaling_factor = 1
    cnt = 0
    obs_list = []
    for lst, height in buildings_xy:
        cnt += 1
        tmp_3d = [(x * scaling_factor, y * scaling_factor, 0) for (x, y) in lst]

        bm = bmesh.new()
        try:
            bm_verts = [bm.verts.new(v) for v in tmp_3d]
            bm_face = bm.faces.new(bm_verts)
        except Exception as e:
            print(f"创建建筑面失败 {cnt}: {e}")
            bm.free()
            continue
            
        bm.normal_update()
        for f in bm.faces:
            if f.normal[2] < 0:
                f.normal_flip()

        me = bpy.data.meshes.new(f"Building_{cnt}")
        bm.to_mesh(me)
        bm.free()
        ob = bpy.data.objects.new(f"Building_{cnt}", me)

        solidify = ob.modifiers.new("Solidify", type='SOLIDIFY')
        solidify.thickness = height * scaling_factor
        solidify.offset = 1

        ob.data.materials.clear()
        ob.data.materials.append(mat_wall_global)
        ob.data.materials.append(mat_roof_global)

        me.calc_normals_split()
        for poly in me.polygons:
            poly.material_index = 1 if poly.normal.z > 0 else 0

        obs_list.append(ob)
        print(f"Building {cnt} done!")

    for ob in obs_list:
        bpy.context.scene.collection.objects.link(ob)
    print("All buildings imported successfully!")

    # 创建地面平面
    bpy.ops.mesh.primitive_plane_add(size=1280, enter_editmode=False, align='WORLD', location=(0, 0, 0))
    plane = bpy.context.object
    plane.name = "Ground_Plane"
    plane.data.materials.append(mat_concrete_global)
    print("Ground plane imported successfully!")


# -------------------------------
# 3. 裁剪超出区域的部分（保留边界内部分）
# -------------------------------
def crop_scene():
    CENTER = mathutils.Vector((0, 0, 0))
    AREA_SIZE = (1280, 1280)  # 区域大小
    half_w, half_h = AREA_SIZE[0]/2, AREA_SIZE[1]/2
    
    # 创建裁剪边界立方体（作为布尔运算的工具）
    bpy.ops.mesh.primitive_cube_add(size=1, location=(CENTER.x, CENTER.y, 0))
    cutter = bpy.context.object
    cutter.name = "Crop_Boundary"
    # 调整立方体大小以匹配区域范围（Z轴高度足够覆盖建筑物）
    cutter.dimensions = (AREA_SIZE[0], AREA_SIZE[1], 1000)  # Z轴高度设为1000米（足够高）
    cutter.hide_viewport = True  # 隐藏裁剪工具（不影响最终渲染）

    # 获取所有建筑物
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH' and obj.name != "Ground_Plane" and obj.name != "Crop_Boundary"]
    deleted_count = 0
    cropped_count = 0

    for obj in mesh_objects:
        # 检查建筑物是否完全在边界外（完全在外部则直接删除）
        all_outside = True
        for v in obj.data.vertices:
            global_v = obj.matrix_world @ v.co
            if -half_w <= global_v.x <= half_w and -half_h <= global_v.y <= half_h:
                all_outside = False
                break
        
        if all_outside:
            bpy.data.objects.remove(obj, do_unlink=True)
            deleted_count += 1
            continue

        # 检查建筑物是否部分在边界内（需要裁剪）
        has_outside = False
        for v in obj.data.vertices:
            global_v = obj.matrix_world @ v.co
            if global_v.x < -half_w or global_v.x > half_w or global_v.y < -half_h or global_v.y > half_h:
                has_outside = True
                break
        
        if has_outside:
            # 添加布尔修改器（用边界立方体切割建筑物）
            bool_mod = obj.modifiers.new(name="Crop", type='BOOLEAN')
            bool_mod.operation = 'INTERSECT'  # 保留与边界的交集部分（即边界内的部分）
            bool_mod.object = cutter
            # 应用修改器
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.modifier_apply(modifier=bool_mod.name)
            cropped_count += 1

    # 删除裁剪工具
    bpy.data.objects.remove(cutter, do_unlink=True)
    
    print(f"删除了 {deleted_count} 个完全在边界外的建筑物")
    print(f"裁剪了 {cropped_count} 个部分超出边界的建筑物（保留边界内部分）")


# -------------------------------
# 4. 导出为 XML 文件
# -------------------------------
def export_xml(export_filepath):
    bpy.ops.export_scene.mitsuba(
        filepath=export_filepath,
        axis_forward='Y',
        axis_up='Z',
        export_ids=True,
        use_selection=False,
        split_files=False,
        ignore_background=True
    )
    print(f"Scene exported to XML: {export_filepath}")


# -------------------------------
# 处理流程与主函数
# -------------------------------
def process_file(input_file, output_file):
    print(f"Processing input: {input_file}")
    clear_scene()
    import_osm(input_file)
    crop_scene()
    export_xml(output_file)
    print("-" * 50)


def main():
    input_dir = "/Users/xiaojieli/Downloads/SEU_Research_XLMIMO_RadioMap/Dataset_Test/dataset"
    output_base_dir = "/Users/xiaojieli/Downloads/SEU_Research_XLMIMO_RadioMap/Dataset_Test/sionnamaps"

    input_files = sorted(glob.glob(os.path.join(input_dir, "*.osm")))
    if not input_files:
        print("未找到任何 OSM 文件！")
        return

    for idx, input_file in enumerate(input_files, start=1):
        output_folder = os.path.join(output_base_dir, f"u{idx}")
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, f"u{idx}.xml")
        process_file(input_file, output_file)


if __name__ == "__main__":
    main()
    