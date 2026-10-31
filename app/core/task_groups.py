"""Keep historical model-specific task keys in their original leaf group."""
from sqlalchemy import or_

LEAF_TASKS = ("leaf", "leaf_our", "leaf_v11", "leaf_v11_head", "leaf_v11_p2",
              "leaf_v11_p2_head", "leaf_asf", "leaf_svbdet", "leaf_sod", "leaf_subtle")


def task_group(value):
    return "leaf" if str(value or "").strip().lower() in LEAF_TASKS else "stem"


def group_condition(column, group):
    return column.in_(LEAF_TASKS) if task_group(group) == "leaf" else or_(column == "stem", column.is_(None))
