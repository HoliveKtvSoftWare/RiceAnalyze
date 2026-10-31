"""剑叶 Excel 导出服务。

复用茎秆导出的落盘 / 列宽 / 文件命名逻辑，只替换两处：

1. 可导出列 —— 包含实际侧脉长度，暂不提供排列顺序（见 ``leaf_analysis.LEAF_COLUMNS``）；
2. 指标口径 —— ``leaf_analysis.compute_from_paths``（主脉/侧脉/空腔 + 7 类维管束）。

因此 ``load_analysis_data_from_record``、``generate_excel_data``、
``create_excel_file``、``export_all_to_excel`` 全部继承茎秆实现，
不需要在路由层区分两套导出代码。
"""

from typing import Any, Dict

from app.services import leaf_analysis
from app.services.stem_excel import ExcelDownloadService


class LeafExcelService(ExcelDownloadService):
    def __init__(self):
        super().__init__()
        # 覆盖父类的茎秆列定义
        self.available_columns = dict(leaf_analysis.LEAF_COLUMNS)

    def _compute_from_paths(self, original_file_path: str, json_file_path: str,
                            unit: str, scale: float) -> Dict[str, Any]:
        return leaf_analysis.compute_from_paths(original_file_path, json_file_path, unit, scale)


leaf_excel_service = LeafExcelService()

__all__ = ["LeafExcelService", "leaf_excel_service"]
