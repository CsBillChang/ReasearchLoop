"""订阅输入与日历计算；所有持久时刻为UTC，调度日按订阅时区计算。"""
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from pydantic import BaseModel, Field, field_validator
from research_loop.providers.paper_sources import DEFAULT_SEARCH_SOURCES, SUPPORTED_SOURCES


# 单用户订阅配置；数量有硬上限，临时聊天不修改本结构。
class SubscriptionInput(BaseModel):
    title: str = Field(default='每日科研简报', min_length=1, max_length=80)
    directions: list[str] = Field(min_length=1, max_length=3)
    sources: list[str] = Field(default_factory=lambda: list(DEFAULT_SEARCH_SOURCES), min_length=1, max_length=17)
    timezone: str = 'Asia/Shanghai'
    at: str = '09:00'
    limit: int = Field(default=5, ge=1, le=5)
    lookback_days: int = Field(default=7, ge=1, le=365)
    enabled: bool = False
    use_profile: bool = True
    expected_revision: int = Field(default=0, ge=0)
    idempotency_key: str = Field(min_length=1, max_length=100)

    @field_validator('directions')
    @classmethod
    def valid_directions(cls, values):
        """拒绝空查询并去重；不将任意画像文本自动扩为新查询。"""
        if any(not x.strip() or len(x) > 200 for x in values):
            raise ValueError('方向必须为1至200字符')
        return list(dict.fromkeys(x.strip() for x in values))

    @field_validator('sources')
    @classmethod
    def valid_sources(cls, values):
        """只接受论文来源目录，不允许输入任意接口地址。"""
        if set(values) - set(SUPPORTED_SOURCES):
            raise ValueError('未知论文来源')
        return list(dict.fromkeys(values))

    @field_validator('timezone')
    @classmethod
    def valid_timezone(cls, value):
        """使用IANA时区；本机没有该时区时拒绝配置。"""
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError("未知IANA时区") from exc
        return value

    @field_validator('at')
    @classmethod
    def valid_at(cls, value):
        """只允许HH:MM，避免秒级高频调度。"""
        if len(value) != 5 or time.fromisoformat(value).isoformat(timespec='minutes') != value:
            raise ValueError('时间必须为HH:MM')
        return value


# 用户主动标记与页面曝光分别提交，不相互推导。
class CandidateReadInput(BaseModel):
    identity: str = Field(pattern=r'^[0-9a-f]{64}$')
    read: bool


def now_utc() -> datetime:
    """返回带时区的当前时刻。"""
    return datetime.now(timezone.utc)


def schedule_slot(config: dict, now: datetime, *, following: bool = False) -> datetime:
    """返回最近已到期或下一日历调度点，避免用固定24小时跨越夏令时。

    Args:
        config: 含timezone与at的已校验配置。
        now: 必须带时区的基准时刻。
        following: 为True时返回严格晚于now的下次调度。

    Returns:
        UTC调度时刻。春季不存在的本地时刻顺延，秋季重复时刻只执行一次。
    """
    if now.tzinfo is None:
        raise ValueError('timezone-aware datetime required')
    zone = ZoneInfo(config['timezone'])
    local = now.astimezone(zone)
    day = local.date()
    def slot(d):
        return datetime.combine(d, time.fromisoformat(config['at']), zone).astimezone(timezone.utc)
    value = slot(day)
    if following:
        return value if value > now else slot(day + timedelta(days=1))
    return value if value <= now else slot(day - timedelta(days=1))
