"""合成DNS轮换与连接权限边界；不查询DNS、不建立网络连接。"""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import socket
from threading import Barrier
import unittest
from unittest.mock import Mock


PATH = Path(__file__).resolve().parents[2] / 'scripts/harness_ab_network.py'
SPEC = importlib.util.spec_from_file_location('harness_ab_network', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
OfficialNetworkPolicy = MODULE.OfficialNetworkPolicy


def ipv4(ip, port=443):
    """构造CPython解析器的标准IPv4答案形状。"""
    return (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', (ip, port))


def ipv6(ip, port=443):
    """构造CPython解析器的标准IPv6答案形状。"""
    return (socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, '', (ip, port, 0, 0))


class HarnessAbNetworkTests(unittest.TestCase):
    """验证真实DNS授权前后顺序，而非把静态清单当永久公网许可。"""

    def policy(self, mode='execute', vector_port=6365, initial_ips=None):
        return OfficialNetworkPolicy(mode, vector_port, initial_ips or ['103.220.64.100', '113.125.206.140'])

    def test_initial_manifest_ips_and_arbitrary_public_ips_cannot_connect_before_resolution(self):
        policy = self.policy()
        for ip in ['103.220.64.100', '113.125.206.140', '8.8.8.8']:
            self.assertFalse(policy.allow_connect((ip, 443)))
        snapshot = policy.snapshot()
        self.assertEqual(snapshot['approved_official_ips'], [])
        self.assertEqual(snapshot['initial_manifest_ips'], ['103.220.64.100', '113.125.206.140'])
        self.assertFalse(snapshot['initial_manifest_ips_authorize_connect'])

    def test_rotated_official_answers_authorize_new_addresses_only_after_each_resolver_returns(self):
        policy = self.policy()
        for ip in ['8.8.8.8', '1.1.1.1', '103.220.64.100']:
            def resolve(host, port):
                self.assertEqual((host, port), ('api.deepseek.com', 443))
                self.assertFalse(policy.allow_connect((ip, 443)))
                return [ipv4(ip)]
            policy.resolve(resolve, 'api.deepseek.com', 443)
            self.assertTrue(policy.allow_connect((ip, 443)))
        self.assertTrue(policy.allow_connect(('8.8.8.8', 443)))
        self.assertTrue(policy.allow_connect(('1.1.1.1', 443)))
        self.assertFalse(policy.allow_connect(('9.9.9.9', 443)))
        self.assertEqual(len(policy.snapshot()['dns_events']), 3)

    def test_query_approval_alone_and_new_process_policy_never_grant_connection(self):
        policy = self.policy()
        self.assertTrue(policy.allow_dns('api.deepseek.com', 443))
        self.assertFalse(policy.allow_connect(('8.8.8.8', 443)))
        policy.resolve(lambda *a: [ipv4('8.8.8.8')], 'api.deepseek.com', 443)
        another = self.policy(initial_ips=policy.snapshot()['approved_official_ips'])
        self.assertFalse(another.allow_connect(('8.8.8.8', 443)))

    def test_unapproved_hosts_ports_and_implicit_hostname_connect_are_rejected(self):
        policy = self.policy()
        resolver = Mock(return_value=[ipv4('8.8.8.8')])
        for host, port in [('example.com', 443), ('api.deepseek.com.evil.invalid', 443),
                ('api.deepseek.com.', 443), ('api.deepseek.com', 80),
                ('api.deepseek.com', 8443), ('8.8.8.8', 443), ('api.deepseek.com', True),
                ('api.deepseek.com', '443'), (None, 443)]:
            with self.subTest(host=host, port=port):
                self.assertFalse(policy.allow_dns(host, port))
                with self.assertRaises(PermissionError):
                    policy.resolve(resolver, host, port)
        resolver.assert_not_called()
        self.assertFalse(policy.allow_connect(('api.deepseek.com', 443)))

    def test_public_ipv6_and_ascii_host_bytes_are_supported(self):
        policy = self.policy()
        ip = '2001:4860:4860::8888'
        answers = [ipv6(ip)]
        self.assertIs(policy.resolve(lambda *a: answers, b'API.DEEPSEEK.COM', 443), answers)
        self.assertTrue(policy.allow_connect((ip, 443, 0, 0)))
        self.assertFalse(policy.allow_connect((ip, 443, 0, 1)))
        self.assertFalse(policy.allow_connect((ip + '%en0', 443, 0, 0)))
        self.assertFalse(policy.allow_connect((ip, 80, 0, 0)))

    def test_private_reserved_link_local_and_multicast_answers_reject_entire_query(self):
        denied = ['127.0.0.1', '10.0.0.1', '172.16.0.1', '192.168.0.1',
            '169.254.169.254', '100.64.0.1', '192.0.2.1', '198.51.100.1',
            '203.0.113.1', '224.0.0.1', '240.0.0.1', '0.0.0.0',
            '::1', 'fc00::1', 'fe80::1', 'ff02::1', '2001:db8::1', '::']
        for ip in denied:
            with self.subTest(ip=ip):
                policy = self.policy()
                answer = ipv6(ip) if ':' in ip else ipv4(ip)
                with self.assertRaises(PermissionError):
                    policy.resolve(lambda *a: [ipv4('8.8.8.8'), answer], 'api.deepseek.com', 443)
                self.assertFalse(policy.allow_connect(('8.8.8.8', 443)))
                self.assertFalse(policy.allow_connect((ip, 443)))
                self.assertEqual(policy.snapshot()['approved_official_ips'], [])
                self.assertEqual(policy.snapshot()['dns_events'][0]['status'], 'rejected')

    def test_resolver_failure_is_propagated_and_never_authorizes_historical_ips(self):
        policy = self.policy()
        error = socket.gaierror('synthetic DNS failure')
        resolver = Mock(side_effect=error)
        with self.assertRaises(socket.gaierror) as caught:
            policy.resolve(resolver, 'api.deepseek.com', 443)
        self.assertIs(caught.exception, error)
        self.assertFalse(policy.allow_connect(('103.220.64.100', 443)))
        snapshot = policy.snapshot()
        self.assertEqual(snapshot['dns_events'][0]['status'], 'resolver_error')
        self.assertEqual(snapshot['approved_official_ips'], [])

    def test_empty_malformed_family_or_changed_port_answers_do_not_authorize(self):
        invalid = [[], [('bad',)], [ipv4('8.8.8.8', 80)],
            [(socket.AF_UNIX, socket.SOCK_STREAM, 0, '', '/tmp/synthetic')],
            [(socket.AF_INET, socket.SOCK_STREAM, 0, '', ('8.8.8.8', 443, 0, 0))],
            [(socket.AF_INET6, socket.SOCK_STREAM, 0, '', ('2001:4860:4860::8888', 443))],
            [(socket.AF_INET, socket.SOCK_STREAM, 0, '', ('2001:4860:4860::8888', 443))],
            [(socket.AF_INET6, socket.SOCK_STREAM, 0, '', ('2001:4860:4860::8888', 443, 0, 1))]]
        for answers in invalid:
            with self.subTest(answers=answers):
                policy = self.policy()
                with self.assertRaises(PermissionError):
                    policy.resolve(lambda *a: answers, 'api.deepseek.com', 443)
                self.assertEqual(policy.snapshot()['approved_official_ips'], [])

    def test_qualify_denies_official_queries_and_connects_even_with_initial_manifest(self):
        policy = self.policy(mode='qualify')
        resolver = Mock(return_value=[ipv4('8.8.8.8')])
        self.assertFalse(policy.allow_dns('api.deepseek.com', 443))
        with self.assertRaises(PermissionError):
            policy.resolve(resolver, 'api.deepseek.com', 443)
        resolver.assert_not_called()
        self.assertFalse(policy.allow_connect(('8.8.8.8', 443)))
        self.assertFalse(policy.allow_connect(('103.220.64.100', 443)))

    def test_only_explicit_vector_loopback_port_is_allowed_in_both_modes(self):
        for mode in ['qualify', 'execute']:
            for port in [6365, 6367]:
                with self.subTest(mode=mode, port=port):
                    policy = self.policy(mode=mode, vector_port=port)
                    for host in ['127.0.0.1', '::1', 'localhost']:
                        self.assertTrue(policy.allow_dns(host, port))
                    self.assertTrue(policy.allow_connect(('127.0.0.1', port)))
                    self.assertTrue(policy.allow_connect(('::1', port, 0, 0)))
                    for forbidden in [6339, 6333, 33316, 6365 if port == 6367 else 6367, 443]:
                        self.assertFalse(policy.allow_connect(('127.0.0.1', forbidden)))
                        self.assertFalse(policy.allow_dns('localhost', forbidden))
                    self.assertFalse(policy.allow_connect(('127.0.0.2', port)))
                    policy.resolve(lambda *a: [ipv4('127.0.0.1', port)], 'localhost', port)
                    self.assertEqual(policy.snapshot()['approved_official_ips'], [])

    def test_localhost_rebinding_to_remote_or_other_port_is_rejected(self):
        policy = self.policy(mode='qualify')
        for answers in [[ipv4('8.8.8.8', 6365)], [ipv4('127.0.0.1', 6339)]]:
            with self.subTest(answers=answers):
                with self.assertRaises(PermissionError):
                    policy.resolve(lambda *a: answers, 'localhost', 6365)
        self.assertEqual(policy.snapshot()['approved_official_ips'], [])

    def test_original_resolver_arguments_order_and_answers_are_preserved(self):
        policy = self.policy()
        answers = [ipv6('2001:4860:4860::8888'), ipv4('1.1.1.1'), ipv4('8.8.8.8')]
        resolver = Mock(return_value=answers)
        result = policy.resolve(resolver, 'api.deepseek.com', 443,
            socket.AF_UNSPEC, socket.SOCK_STREAM, proto=socket.IPPROTO_TCP, flags=0)
        self.assertIs(result, answers)
        resolver.assert_called_once_with('api.deepseek.com', 443,
            socket.AF_UNSPEC, socket.SOCK_STREAM, proto=socket.IPPROTO_TCP, flags=0)
        self.assertEqual(policy.snapshot()['dns_events'][0]['raw_answers'], answers)
        result.append(ipv4('9.9.9.9'))
        self.assertFalse(policy.allow_connect(('9.9.9.9', 443)))
        self.assertEqual(len(policy.snapshot()['dns_events'][0]['raw_answers']), 3)

    def test_snapshot_is_json_serializable_and_cannot_mutate_permissions(self):
        policy = self.policy()
        policy.resolve(lambda *a: [ipv4('8.8.8.8')], 'api.deepseek.com', 443)
        snapshot = policy.snapshot()
        json.dumps(snapshot)
        snapshot['approved_official_ips'].append('9.9.9.9')
        snapshot['dns_events'][0]['raw_answers'].append(ipv4('9.9.9.9'))
        self.assertFalse(policy.allow_connect(('9.9.9.9', 443)))
        self.assertEqual(len(policy.snapshot()['dns_events'][0]['raw_answers']), 1)

    def test_parallel_roles_can_resolve_rotating_public_addresses_without_bypassing_order(self):
        policy = self.policy()
        barrier = Barrier(8)
        def role(index):
            ip = f'8.8.8.{index + 1}'
            def resolver(*args):
                self.assertFalse(policy.allow_connect((ip, 443)))
                barrier.wait(timeout=5)
                return [ipv4(ip)]
            policy.resolve(resolver, 'api.deepseek.com', 443)
            return policy.allow_connect((ip, 443))
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(role, range(8)))
        self.assertTrue(all(results))
        snapshot = policy.snapshot()
        self.assertEqual(len(snapshot['approved_official_ips']), 8)
        self.assertEqual(len(snapshot['dns_events']), 8)
        self.assertEqual({event['sequence'] for event in snapshot['dns_events']}, set(range(1, 9)))
        self.assertTrue(all(event['status'] == 'approved' for event in snapshot['dns_events']))

    def test_invalid_modes_and_implicit_vector_ports_are_rejected(self):
        for mode, port in [('unknown', 6365), ('execute', True), ('execute', '6365'),
                ('qualify', 0), ('qualify', 65536)]:
            with self.subTest(mode=mode, port=port):
                with self.assertRaises(ValueError):
                    self.policy(mode=mode, vector_port=port)


if __name__ == '__main__':
    unittest.main()
