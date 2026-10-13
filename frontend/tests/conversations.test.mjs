/** 日期分组覆盖时区、跨年和无时间数据，不依赖浏览器当前时钟。 */
import test from 'node:test';
import assert from 'node:assert/strict';
import {groupConversations} from '../src/components/conversations.js';

test('按最后活跃日期分组并排序，上海午夜正确归类',()=>{
  const items=[{conversation_id:'old',created_at:'2026-10-09T16:01:00Z'},
    {conversation_id:'recent',created_at:'2026-10-01T00:00:00Z',last_activity_at:'2026-10-10T16:01:00Z'},
    {conversation_id:'same',created_at:'2026-10-10T17:00:00Z'}];
  const original=JSON.stringify(items);
  const groups=groupConversations(items,new Date('2026-10-11T04:00:00Z'),'Asia/Shanghai');
  assert.deepEqual(groups.map(g=>g.label),['今天','昨天']);
  assert.deepEqual(groups[0].items.map(c=>c.conversation_id),['same','recent']);
  assert.equal(JSON.stringify(items),original);
});
test('跨年昨天和未知日期有独立组',()=>{
  const groups=groupConversations([{conversation_id:'unknown',created_at:'bad'},
    {conversation_id:'yesterday',created_at:'2026-12-31T12:00:00Z'},
    {conversation_id:'past',created_at:'2026-10-01T00:00:00Z'}],new Date('2027-01-01T04:00:00Z'),'Asia/Shanghai');
  assert.deepEqual(groups.map(g=>g.label),['昨天','2026 / 10 / 01','日期未知']);
});
test('夏令时切换按日历昨天分组',()=>{
  const groups=groupConversations([{conversation_id:'a',created_at:'2026-03-08T08:00:00Z'}],new Date('2026-03-09T07:30:00Z'),'America/Los_Angeles');
  assert.equal(groups[0].label,'昨天');
});
