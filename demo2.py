import sys
import os

# 确保路径被正确添加
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

# 修复重复引用
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

# 定义文件路径
mesh_path = './res.glb'
image_path = 'assets/demo.png'

# 1. 生成白模 (Shape Generation)
print("正在生成几何模型...")
shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')
mesh_untextured = shape_pipeline(image=image_path)[0]

# === [关键修复] 将生成的模型保存到硬盘，以便下一步读取 ===
print(f"正在保存模型到 {mesh_path} ...")
mesh_untextured.export(mesh_path)
# =======================================================

# 2. 生成贴图 (Texture Painting)
#print("正在生成纹理...")
# 注意：paint_pipeline 需要读取刚刚保存的 mesh_path
#paint_pipeline = Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view=6, resolution=512))
#mesh_textured = paint_pipeline(mesh_path, image_path=image_path)

# 3. (可选) 保存最终带贴图的模型
# paint_pipeline 通常会返回结果，但不一定自动保存覆盖，建议手动保存
#output_textured_path = './res_textured.glb'
#if mesh_textured:
    # 假设 mesh_textured 是一个 mesh 对象，根据具体库的API保存
    # 如果 paint_pipeline 内部已经保存了，可以忽略这一步
    # mesh_textured.export(output_textured_path)
#    print("纹理生成完成！")
