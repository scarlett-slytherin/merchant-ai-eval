import csv
import json
from collections import Counter
from datetime import date
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


BUILD = Path(__file__).resolve().parent
ROOT = BUILD.parent
GPT_RUN_DATE = '2026-11-18'
TOPICS = ('团购', '上架', '核销', '佣金')
ORIGINS = ('公开原题', '公开问题改写', '规则派生')
MODELS = {'gpt': 'GPT（Codex）', 'qwen': '通义千问', 'doubao': '豆包'}
HOLDOUT = set('D04 D06 D10 D12 D15 D22 M02 M04 M09 M14 M18 M25'.split())
CLARIFY = set('P02 D01 D04 D08 D12 D13 D15 D17 D24 D30 D32 D39 D42 D45 M01 M05 M06 M15 M20 M23 M25 M30 M32'.split())
REFUSE = set('P03 P05 D09 D18 M12 M17 M24'.split())
ESCALATE = set('P06 D19 M19 M22 M31 M33'.split())
ROUTES = dict.fromkeys(CLARIFY, '澄清或自查') | dict.fromkeys(REFUSE, '拒绝违规部分') | dict.fromkeys(ESCALATE, '引导自查或转人工')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_csv(path, headers, records):
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(records)


def md_table(headers, rows):
    def cell(value):
        return str(value).replace('|', '｜').replace('\n', ' ')

    return '\n'.join(' | '.join(['', *map(cell, row), '']) for row in [headers, ['---'] * len(headers), *rows])


pilot = read_json(BUILD / 'pilot.json')
pilot_input = '下面有六个相互独立的商家咨询。仅使用题内参考资料，不联网。每题一段，以题号开头，只输出面向商家的回复。\n\n'
pilot_input += '\n\n'.join(f"{c['id']}｜{c['platform']}\n【商家问题】{c['question']}\n【参考资料】《{c['source_title']}》：{c['evidence']}" for c in pilot)
(BUILD / '试测输入.txt').write_text(pilot_input + '\n', encoding='utf-8')

sources_in = ('douyin', 'douyin_extra', 'douyin_tail', 'meituan', 'meituan_extra')
cases = pilot + [case for name in sources_in if (BUILD / (name + '.json')).exists() for case in read_json(BUILD / (name + '.json'))]
assert 56 <= len(cases) <= 100
assert len(cases) == len({case['id'] for case in cases}) == len({case['question'] for case in cases})
case_by_id = {case['id']: case for case in cases}
case_by_id['P06']['source_title'] = '美团商户管理制度（第十三条）'
for case in cases:
    case_id = case['id']
    case['split'] = 'pilot' if case_id.startswith('P') else 'holdout' if case_id in HOLDOUT else 'dev'
    case['route'] = ROUTES.get(case_id, '直接答复')
    assert case['topic'] in TOPICS and case['origin'] in ORIGINS
    assert all(case.get(key) for key in ('platform', 'question', 'expected', 'evidence', 'must', 'red_flags', 'source_title'))
    assert case['source_url'].startswith('https://')
    assert all(isinstance(case[key], list) for key in ('must', 'red_flags'))

(ROOT / 'dataset.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
case_headers = ['题号', '商家问题', '平台', '主题', '题目来源', '集合', '预期行为', '规则依据', '参考答案', '必答点', '错误判据', '出处', '官方链接']
case_rows = [[case['id'], case['question'], case['platform'], case['topic'], case['origin'], case['split'], case['route'], case['evidence'], case['expected'], '／'.join(case['must']), '／'.join(case['red_flags']), case['source_title'], case['source_url']] for case in cases]
write_csv(ROOT / '评测集.csv', case_headers, case_rows)

grade_file = BUILD / 'grades.json'
grades = read_json(grade_file) if grade_file.exists() else {}
metadata_file = BUILD / 'run_metadata.json'
run_metadata = read_json(metadata_file) if metadata_file.exists() else {}
assert isinstance(run_metadata, dict), 'run_metadata.json 需为按运行编号索引的对象'
prompts = read_json(ROOT / 'prompts.json')
raw = []
batch_replies = []
groups = []
for provider, display, split in [(p, d, 'pilot') for p, d in MODELS.items()] + [('gpt', MODELS['gpt'], 'holdout')]:
    for version in prompts:
        key = f'{provider}_{version.lower()}' if split == 'pilot' else f'{provider}_{split}_{version.lower()}'
        path = BUILD / (key + '.json')
        replies = read_json(path) if path.exists() else {}
        split_ids = {case['id'] for case in cases if case['split'] == split}
        assert set(replies).issubset(split_ids)
        batch = None
        if split == 'pilot' and provider == 'qwen':
            batch_path = BUILD / f'{provider}_batch_{version.lower()}.json'
            if batch_path.exists():
                batch = read_json(batch_path)
                assert all(isinstance(batch.get(field), str) and batch[field].strip() for field in
                           ('provider', 'version', 'split', 'date', 'channel', 'model_label', 'outcome', 'reply'))
                assert (batch['provider'], batch['version'], batch['split']) == (provider, version, split)
                assert sorted(batch['case_ids']) == sorted(split_ids)
                assert date.fromisoformat(batch['date']).isoformat() == batch['date']
                batch_replies.append(batch)
        metadata = run_metadata.get(key, {})
        assert isinstance(metadata, dict), f'{key} 的运行信息需为对象'
        run_date = metadata.get('date', GPT_RUN_DATE if provider == 'gpt' else '')
        channel = metadata.get('channel', 'Codex' if provider == 'gpt' else '')
        model_label = metadata.get('model_label', '')
        assert all(isinstance(value, str) for value in (run_date, channel, model_label)), key
        assert not run_date or date.fromisoformat(run_date).isoformat() == run_date, key
        graded = []
        for case_id, reply in sorted(replies.items()):
            assert isinstance(reply, str) and reply.strip()
            grade = grades.get(key, {}).get(case_id)
            entry = dict(provider=provider, model=display, version=version, split=split, case_id=case_id, reply=reply,
                         date=run_date, channel=channel, model_label=model_label)
            if grade:
                a, h, t, r, critical, note = grade
                assert a in (0, 1, 2) and h in (0, 2) and t in (0, 1, 2) and r in (0, 1, 2)
                passed = a == h == r == 2 and t >= 1 and not critical
                entry.update(accuracy=a, grounded=h, tone=t, routing=r, critical=critical, passed=passed, note=note)
                graded.append(entry)
            raw.append(entry)
        n = len(graded)
        status = '已评分' if n == len(replies) == len(split_ids) else '部分完成' if replies else '待实测'
        if batch and not replies:
            status = batch['outcome']
        counts = ['—'] * 6
        if n:
            counts = [f'{sum(x[f] == 2 for x in graded)}/{n}' for f in ('accuracy', 'grounded', 'tone', 'routing')]
            counts.extend([f"{sum(x['critical'] for x in graded)}/{n}", f"{sum(x['passed'] for x in graded)}/{n}"])
        groups.append([display, version, split, status, f'{len(replies)}/{len(split_ids)}', *counts])

with (ROOT / '试测记录.jsonl').open('w', encoding='utf-8') as handle:
    for entry in raw:
        handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
raw_headers = ['题号', '模型', '版本', '集合', '原始输出', '准确性', '事实有据', '话术', '答复边界', '严重错误', '通过', '判分记录', '运行日期', '入口', '模型标识']
raw_rows = [[x['case_id'], x['model'], x['version'], x['split'], x['reply'], x.get('accuracy', ''), x.get('grounded', ''), x.get('tone', ''), x.get('routing', ''), '是' if x.get('critical') else '否' if 'critical' in x else '', '通过' if x.get('passed') else '未通过' if 'passed' in x else '待评分', x.get('note', ''), x['date'], x['channel'], x['model_label']] for x in raw]
write_csv(ROOT / '试测记录.csv', raw_headers, raw_rows)
summary_headers = ['模型', 'Prompt', '集合', '状态', '完成', '准确满分', '有据', '话术满分', '边界满分', '严重错误', '通过']
summary_table = md_table(summary_headers, groups)
write_csv(ROOT / '对比表.csv', summary_headers, groups)

counts = Counter(case['topic'] for case in cases)
origins = Counter(case['origin'] for case in cases)
sources = {case['source_url'] for case in cases}
status = f"共 {len(raw)} 条逐题回复，已评分 {sum('passed' in x for x in raw)} 条。"
pilot_reply_target = len(pilot) * len(prompts)
pending_vendors = []
for provider in ('qwen', 'doubao'):
    vendor_replies = [entry for entry in raw if entry['provider'] == provider and entry['split'] == 'pilot']
    vendor_batches = [entry for entry in batch_replies if entry['provider'] == provider]
    if vendor_batches:
        status += f'{MODELS[provider]}另有 {len(vendor_batches)} 条整批回复。'
    if not vendor_replies:
        if not vendor_batches:
            pending_vendors.append(MODELS[provider])
        continue
    graded_count = sum('passed' in entry for entry in vendor_replies)
    status += f"{MODELS[provider]}已保存 {len(vendor_replies)}/{pilot_reply_target} 条回复，已评分 {graded_count} 条。"
if pending_vendors:
    status += f"待补齐{'和'.join(pending_vendors)}结果。"
latest_run_date = max((entry['date'] for entry in [*raw, *batch_replies] if entry['date']), default=GPT_RUN_DATE)
report_date = date.fromisoformat(latest_run_date)
report_date_label = f'{report_date.year} 年 {report_date.month} 月 {report_date.day} 日'
distribution = '、'.join(f'{topic} {counts[topic]} 条' for topic in TOPICS)
holdout_topics = Counter(case['topic'] for case in cases if case['split'] == 'holdout')
holdout_distribution = '、'.join(f'{topic} {holdout_topics[topic]} 题' for topic in TOPICS)
provenance = '、'.join(f'{kind} {origins[kind]} 条' for kind in ORIGINS)
breakdown = [[f'P0{i}', *(next((x['note'] for x in raw if x['provider'] == 'gpt' and x['version'] == v and x['case_id'] == f'P0{i}' and 'note' in x), '待评分') for v in prompts)] for i in range(1, 7)]
bad_cases = [x for x in raw if x.get('passed') is False]
bad_table = md_table(['模型', '版本', '题号', '问题', '评分说明'], [[x['model'], x['version'], x['case_id'], case_by_id[x['case_id']]['question'], x['note']] for x in bad_cases]) if bad_cases else '已评分回复中暂无判定为未通过的用例。下表为复测时要捕捉的典型错误。'
batch_evidence = []
if batch_replies:
    batch_evidence.extend([
        '### 整批回复记录', '',
        '整批回复单独保留。一次回复未给出独立题号时，该次逐题覆盖为 0/6，评分列留空。', '',
    ])
    for batch in batch_replies:
        batch_evidence.extend([
            f"**{MODELS[batch['provider']]} · {batch['version']}**　{batch['channel']} · {batch['model_label']} · "
            f"{batch['date']} · {batch['case_ids'][0]}–{batch['case_ids'][-1]} · {batch['outcome']}", '',
            *(f'> {line}' for line in batch['reply'].splitlines()), '',
        ])

template = (BUILD / 'report_template.md').read_text(encoding='utf-8')
findings_file = BUILD / 'findings.md'
findings = findings_file.read_text(encoding='utf-8') if findings_file.exists() else '待复核试测输出。'
for name, value in {
    'TOTAL': str(len(cases)), 'DEVCOUNT': str(sum(case['split'] == 'dev' for case in cases)), 'HOLDOUTDIST': holdout_distribution,
    'STATUS': status, 'REPORT_DATE': report_date_label, 'DISTRIBUTION': distribution,
    'PROVENANCE': provenance, 'SOURCECOUNT': str(len(sources)),
    'RESULTS': summary_table, 'BATCH_EVIDENCE': '\n'.join(batch_evidence),
    'BREAKDOWN': md_table(['题号', 'GPT V0', 'GPT V1'], breakdown),
    'BADCASES': bad_table, 'FINDINGS': findings, 'V0': prompts['V0'], 'V1': prompts['V1'],
}.items():
    template = template.replace('{{' + name + '}}', value)
report = ROOT / '本地生活商家咨询 AI 助手评测.md'
report.write_text(template, encoding='utf-8')

details = ['# 评测集明细', '', f'{len(cases)} 条。{distribution}。', '']
for case in cases:
    details.extend([
        f"## {case['id']}｜{case['question']}", '',
        f"{case['platform']} · {case['topic']} · {case['origin']} · {case['split']} · {case['route']}", '',
        f"**规则依据**　{case['evidence']}", '',
        f"**参考答案**　{case['expected']}", '',
        f"**必答点**　{'；'.join(case['must'])}", '',
        f"**错误判据**　{'；'.join(case['red_flags'])}", '',
        f"[{case['source_title']}]({case['source_url']})", '',
    ])
(BUILD / '评测集明细.md').write_text('\n'.join(details), encoding='utf-8')
with ZipFile(ROOT / '本地生活商家咨询评测·Notion导入包.zip', 'w', ZIP_DEFLATED) as archive:
    for path in (report, ROOT / '评测集.csv', ROOT / '对比表.csv', ROOT / '试测记录.csv'):
        archive.write(path, arcname=f'本地生活商家咨询评测/{path.name}')

print(json.dumps({'cases': len(cases), 'splits': dict(Counter(c['split'] for c in cases)), 'topics': counts,
                  'origins': origins, 'sources': len(sources), 'replies': len(raw), 'batch_replies': len(batch_replies),
                  'groups': groups}, ensure_ascii=False))
