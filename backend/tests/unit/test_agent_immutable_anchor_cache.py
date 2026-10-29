"""封存历史context缓存的通用攻击回归；不访问真实旧锚或网络。"""
from copy import deepcopy
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
import agent_final_completion as c


def write(path,value):
    path.write_text(json.dumps(value,sort_keys=True)+'\n')


class ImmutableAnchorCacheTests(unittest.TestCase):
    def setUp(self):
        c.clear_validation_caches();self.addCleanup(c.clear_validation_caches)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name).resolve();self.root=self.base/'sealed';self.root.mkdir()
        self.file=self.root/'payload.txt';self.file.write_text('original payload')
        self.auth=self.root/'closed-auth.json';write(self.auth,{'approved':False,'allowed_modes':[]})
        self.external=self.base/'external-proof.json';write(self.external,{'status':'passed'})
        self.seal();self.calls=0

    def seal(self):
        rows={str(p.relative_to(self.root)):{'sha256':c._helpers()._hash(p),'bytes':p.stat().st_size}
              for p in self.root.rglob('*') if p.is_file() and p.name not in {'SEAL.json','SEAL.sha256','SEAL-READBACK.json'}}
        write(self.root/'SEAL.json',{'files':rows});value=c._helpers()._hash(self.root/'SEAL.json')
        (self.root/'SEAL.sha256').write_text(value+'  SEAL.json\n')
        write(self.root/'SEAL-READBACK.json',{'status':'passed','seal_sha256':value})

    def validate(self):
        self.calls+=1;root,_,digest=c._sealed(self.root)
        auth=json.loads(self.auth.read_text());c._require(auth=={'approved':False,'allowed_modes':[]},'CLOSED_AUTH_REQUIRED')
        c._verified_hash(self.external);c._require(json.loads(self.external.read_text())=={'status':'passed'},'EXTERNAL_PROOF_REQUIRED')
        return {'anchor_seals':{str(root):digest},'rows':[{'status':'partial','quality':None}]}

    def build(self):return c._cached_anchor('synthetic-test',self.root,frozenset(),self.validate)

    def test_hot_context_is_deepcopy_and_does_not_rewalk_seal_or_read_bytes(self):
        expected=self.build();first=self.build();first['rows'][0]['quality']='full'
        with patch.object(c,'_sealed',side_effect=AssertionError('no full walk on unchanged dependency')):
            self.assertEqual(self.build(),expected)
        self.assertEqual(self.calls,1);self.assertEqual(c._VALIDATION_CACHE_COUNTS['hits'],2)

    def test_same_size_byte_change_with_restored_mtime_rejects_by_ctime(self):
        self.build();previous=self.file.stat();self.file.write_text('modified payload')
        os.utime(self.file,ns=(previous.st_atime_ns,previous.st_mtime_ns))
        with self.assertRaises(PermissionError):self.build()
        self.assertNotIn(('synthetic-test',str(self.root)),c._IMMUTABLE_CONTEXTS)

    def test_new_deleted_renamed_and_symlinked_paths_never_keep_cached_value(self):
        for mutation in ('new','deleted','renamed','symlink'):
            with self.subTest(mutation=mutation):
                c.clear_validation_caches();self.build()
                original=self.file.read_bytes()
                if mutation=='new':(self.root/'extra').write_text('extra')
                elif mutation=='deleted':self.file.unlink()
                elif mutation=='renamed':self.file.rename(self.root/'renamed')
                else:self.file.unlink();self.file.symlink_to(self.external)
                with self.assertRaises(PermissionError):self.build()
                if mutation=='new':(self.root/'extra').unlink()
                elif mutation=='renamed':(self.root/'renamed').unlink()
                elif mutation=='symlink':self.file.unlink()
                self.file.write_bytes(original)

    def test_equal_bytes_inode_replacement_requires_full_validation(self):
        self.build();before=self.calls;replacement=self.root/'replacement';replacement.write_bytes(self.file.read_bytes())
        os.replace(replacement,self.file);self.build();self.assertEqual(self.calls,before+1)

    def test_hardlink_mode_and_external_proof_changes_invalidate(self):
        self.build();os.link(self.file,self.base/'hardlink')
        before=self.calls;self.build();self.assertEqual(self.calls,before+1)
        self.file.chmod(0o600);before=self.calls;self.build();self.assertEqual(self.calls,before+1)
        write(self.external,{'status':'failed'})
        with self.assertRaises(PermissionError):self.build()

    def test_seal_readback_closed_auth_and_parent_proof_changes_are_not_cached(self):
        for path,change in ((self.root/'SEAL-READBACK.json',{'status':'failed'}),(self.auth,{'approved':True})):
            with self.subTest(path=path.name):
                c.clear_validation_caches();self.build();before=path.read_bytes();value=json.loads(before);value.update(change);write(path,value)
                with self.assertRaises(PermissionError):self.build()
                path.write_bytes(before)

    def test_failed_verification_never_publishes_context(self):
        with patch.object(self,'validate',side_effect=PermissionError('synthetic denial')):
            with self.assertRaises(PermissionError):self.build()
        self.assertFalse(c._IMMUTABLE_CONTEXTS)

    def test_sql_cache_clear_and_unified_clear_cannot_hide_cold_miss(self):
        c._ANCHOR_CACHE['synthetic-sql-key']={'known':True};self.build();self.assertEqual(self.calls,1)
        c._ANCHOR_CACHE.clear();self.build();self.assertEqual(self.calls,2)
        c.clear_validation_caches();self.assertFalse(c._IMMUTABLE_CONTEXTS);self.assertFalse(c._ANCHOR_CACHE)
        self.assertFalse(c._helpers()._HASH_CACHE)

    def test_ancestor_symlink_is_rejected_even_when_target_files_are_identical(self):
        self.build();moved=self.base.with_name(self.base.name+'-moved')
        self.base.rename(moved);self.base.symlink_to(moved,target_is_directory=True)
        try:
            with self.assertRaises(PermissionError):self.build()
        finally:
            self.base.unlink();moved.rename(self.base)

    def test_unrelated_system_temp_entries_do_not_invalidate_immutable_dependencies(self):
        self.build()
        with tempfile.TemporaryDirectory():self.build()
        self.assertEqual(self.calls,1)

    def test_concurrent_readers_only_publish_one_validated_context(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as pool:
            values=list(pool.map(lambda _:self.build(),range(12)))
        self.assertTrue(all(value==values[0] for value in values));self.assertEqual(self.calls,1)
        values[0]['rows'].clear();self.assertTrue(self.build()['rows'])

    def test_cycle_check_remains_before_hot_cache_and_live_auth_is_not_stored(self):
        self.build()
        with self.assertRaises(PermissionError):c._cached_anchor('synthetic-test',self.root,{str(self.root)},self.validate)
        live=self.base/'live-auth.json';write(live,{'approved':True});self.build()
        # 活授权由调用方实时读取；不作为缓存中已批准的状态。
        write(live,{'approved':False});self.assertFalse(json.loads(live.read_text())['approved'])
        self.assertNotIn(str(live),next(iter(c._IMMUTABLE_CONTEXTS.values()))['capture']['paths'])


if __name__=='__main__':unittest.main(verbosity=2)
