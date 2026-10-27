"""H1实验的官方域名DNS授权；不发请求、不修改TLS或业务工具配置。"""
from __future__ import annotations

from copy import deepcopy
import ipaddress
import socket
from threading import RLock
import time


class OfficialNetworkPolicy:
    """仅将本进程本次官方域名解析的公开地址授予443连接权限。"""

    OFFICIAL_HOST = 'api.deepseek.com'
    LOOPBACK_HOSTS = frozenset({'127.0.0.1', '::1', 'localhost'})
    LOOPBACK_IPS = frozenset({'127.0.0.1', '::1'})

    def __init__(self, mode, vector_port, initial_ips):
        """登记运行模式和向量端口；历史清单IP只留证，不提供连接授权。"""
        if mode not in {'qualify', 'execute'}:
            raise ValueError('H1 network mode must be qualify or execute')
        if isinstance(vector_port, bool) or not isinstance(vector_port, int) or not 1 <= vector_port <= 65535:
            raise ValueError('H1 vector port must be an explicit valid integer')
        self.mode, self.vector_port = mode, vector_port
        self._lock = RLock()
        self._initial_ips = list(initial_ips or [])
        self._approved_ips = set()
        self._events = []
        self._sequence = 0

    @staticmethod
    def _host(host):
        """DNS名称不区分ASCII大小写；拒绝隐式别名或非ASCII编码。"""
        if isinstance(host, bytes):
            try:
                host = host.decode('ascii')
            except UnicodeDecodeError:
                return None
        return host.lower() if isinstance(host, str) else None

    @staticmethod
    def _port(port):
        return port if isinstance(port, int) and not isinstance(port, bool) else None

    @staticmethod
    def _ip(value):
        """统一数字地址表达，不接受作用域后缀引入隐式接口授权。"""
        if not isinstance(value, str) or '%' in value:
            raise ValueError('H1 numeric IP address required')
        return ipaddress.ip_address(value)

    @staticmethod
    def _public(ip):
        """global之外再排除多播、保留段等不能充当普通公网目标的地址。"""
        return (ip.is_global and not ip.is_private and not ip.is_loopback
                and not ip.is_link_local and not ip.is_multicast
                and not ip.is_reserved and not ip.is_unspecified)

    def allow_dns(self, host, port):
        """供审计守卫检查查询入口，批准查询本身不会批准任何远端IP。"""
        host, port = self._host(host), self._port(port)
        with self._lock:
            return ((host in self.LOOPBACK_HOSTS and port == self.vector_port)
                    or (self.mode == 'execute' and host == self.OFFICIAL_HOST and port == 443))

    def allow_connect(self, address):
        """连接必须为已解析数字地址；不允许connect内部再次按域名隐式解析。"""
        if not isinstance(address, (tuple, list)) or len(address) not in {2, 4}:
            return False
        try:
            ip, port = self._ip(address[0]), self._port(address[1])
        except ValueError:
            return False
        if len(address) == 4 and (not isinstance(ip, ipaddress.IPv6Address)
                or address[2] != 0 or address[3] != 0):
            return False
        with self._lock:
            if str(ip) in self.LOOPBACK_IPS and port == self.vector_port:
                return True
            return (self.mode == 'execute' and port == 443
                    and self._public(ip) and str(ip) in self._approved_ips)

    def resolve(self, original_resolver, host, port, *args, **kwargs):
        """保留原解析结果；全部答案通过检查后才原子授予连接权限。

        Args:
            original_resolver: 未被替换的socket.getaddrinfo或测试内存解析器。
            host: 调用者实际请求解析的名称。
            port: 官方443或本批专属向量端口。
            *args: 原样传给原解析器的位置参数。
            **kwargs: 原样传给原解析器的关键字参数。

        Returns:
            原解析器返回的原结果，不改地址顺序或socket行为。

        Raises:
            PermissionError: 目标超出范围，或解析混入非公开/错误端口答案。
        """
        normalized_host, normalized_port = self._host(host), self._port(port)
        with self._lock:
            self._sequence += 1
            event = {'sequence': self._sequence, 'at_unix': time.time(),
                'host': normalized_host, 'port': normalized_port,
                'status': 'resolving', 'raw_answers': None, 'approved_ips': []}
            self._events.append(event)
            if not self.allow_dns(host, port):
                event.update(status='rejected', reason='query_outside_frozen_scope')
                raise PermissionError('H1 DNS query outside frozen scope')
        try:
            answers = original_resolver(host, port, *args, **kwargs)
        except BaseException as exc:
            with self._lock:
                event.update(status='resolver_error', error_type=type(exc).__name__)
            raise
        with self._lock:
            # 捕获调用方后来修改结果前的原始答案，不以它替换原返回值。
            event['raw_answers'] = deepcopy(answers)
        ips = set()
        try:
            if not isinstance(answers, (list, tuple)) or not answers:
                raise ValueError('resolver returned no usable address records')
            official = normalized_host == self.OFFICIAL_HOST
            for answer in answers:
                if not isinstance(answer, (tuple, list)) or len(answer) != 5:
                    raise ValueError('resolver returned malformed address record')
                family, _, _, _, address = answer
                if family not in {socket.AF_INET, socket.AF_INET6}:
                    raise ValueError('resolver returned unsupported address family')
                expected_length = 2 if family == socket.AF_INET else 4
                if not isinstance(address, (tuple, list)) or len(address) != expected_length:
                    raise ValueError('resolver returned malformed socket address')
                ip = self._ip(address[0])
                if self._port(address[1]) != normalized_port:
                    raise ValueError('resolver answer port differs from approved query')
                if ((family == socket.AF_INET) != isinstance(ip, ipaddress.IPv4Address)
                        or (len(address) == 4 and (address[2] != 0 or address[3] != 0))):
                    raise ValueError('resolver address family or interface differs from scope')
                if official:
                    if not self._public(ip):
                        raise ValueError('official DNS returned a non-public address')
                elif str(ip) not in self.LOOPBACK_IPS:
                    raise ValueError('loopback DNS returned a non-loopback address')
                ips.add(str(ip))
        except (TypeError, ValueError) as exc:
            with self._lock:
                event.update(status='rejected', reason=str(exc))
            raise PermissionError('H1 DNS answer outside frozen scope: ' + str(exc)) from exc
        with self._lock:
            if normalized_host == self.OFFICIAL_HOST:
                self._approved_ips.update(ips)
            event.update(status='approved', approved_ips=sorted(ips))
        return answers

    def snapshot(self):
        """返回可序列化且与内部状态解耦的DNS授权记录。"""
        with self._lock:
            return deepcopy({'mode': self.mode, 'official_host': self.OFFICIAL_HOST,
                'official_port': 443, 'vector_port': self.vector_port,
                'initial_manifest_ips': self._initial_ips,
                'initial_manifest_ips_authorize_connect': False,
                'approved_official_ips': sorted(self._approved_ips),
                'dns_events': self._events})
