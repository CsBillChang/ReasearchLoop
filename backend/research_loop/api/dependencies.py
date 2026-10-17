"""HTTP 依赖装配：统一配置、数据库和服务构造，路由不相互借用私有工厂。"""
from fastapi import Depends

from research_loop.config import Settings, get_settings
from research_loop.services.artifacts import ArtifactService
from research_loop.services.note_versions import NoteVersionService
from research_loop.services.tasks import TaskService
from research_loop.storage.files import FileStore
from research_loop.storage.mysql import Database


def get_runtime_settings() -> Settings:
    """读取本次请求配置；测试可通过 FastAPI dependency_overrides 注入隔离配置。"""
    return get_settings()


def get_database(settings: Settings = Depends(get_runtime_settings)) -> Database:
    """构造请求内共享的数据库适配器，保留现有基础表就绪检查。

    Returns:
        使用当前配置的数据库；不持有跨请求连接，也不构造模型客户端。
    """
    db = Database(settings.database_url)
    db.ensure_schema()
    return db


def get_task_service(
    db: Database = Depends(get_database),
    settings: Settings = Depends(get_runtime_settings),
) -> TaskService:
    """装配任务服务；请求策略使用与数据库相同的配置快照。"""
    return TaskService(db, settings)


def get_artifact_service(
    db: Database = Depends(get_database),
    settings: Settings = Depends(get_runtime_settings),
) -> ArtifactService:
    """装配成果查询和保存服务，文件根目录只从后端配置取得。"""
    return ArtifactService(db, settings)


def get_note_version_service(
    db: Database = Depends(get_database),
    settings: Settings = Depends(get_runtime_settings),
) -> NoteVersionService:
    """装配项目笔记版本服务；同一请求复用数据库和配置依赖。"""
    files = FileStore(settings.data_dir, settings.output_dir, settings.project_root)
    return NoteVersionService(db, files)
