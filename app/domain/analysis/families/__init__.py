# -*- coding: utf-8 -*-
"""分析大类的指标算法。

每个族一个模块，导出统一的 ``compute(analysis_data, unit, scale) -> dict``：
``analysis_data`` 是 LabelMe 结果（``{"shapes": [...]}``），返回值是
``{列 key: 数值}``，列 key 与该族 ``features/task_catalog/families/<族>.py``
里的列定义一一对应。

加一个新族 = 在本目录加一个模块 + 在 ``features/task_catalog/families/``
加对应的族定义，共享核心无需改动。
"""
from . import leaf

__all__ = ["leaf"]
