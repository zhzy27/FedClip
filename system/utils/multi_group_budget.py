"""Read-only parameter audit for an independent experiment, never a training hook.

Counts refer to deduplicated Parameter objects, including biases and fc3. They
are neither peak memory nor measured network traffic. No default rank is changed.
"""
import json
import math
from pathlib import Path

from common.experiment import rng_state, set_rng, sha256


DEFAULT_CONFIG = Path(__file__).resolve().parents[1]/'experiments/multi_group_budget_v1.json'


def read_config(path=DEFAULT_CONFIG):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get('schema_version') != 1 or value.get('assignment') != 'client_id_modulo_levels':
        raise ValueError('Unsupported parameter-budget configuration')
    rates = value['rank_rates']
    if not rates or any(not math.isfinite(r) or not 0 < r < 1 for r in rates):
        raise ValueError('Budget rates must be finite and in (0,1)')
    if any(type(value[key]) is not int or value[key] < 1 for key in ('num_clients','num_server_models')):
        raise ValueError('Budget N and K must be positive integers')
    for key in ('baseline_classes','expected_baseline_parameters','expected_single_parameters','expected_total_parameters'):
        if len(value[key]) != len(rates):
            raise ValueError(f'Budget table length differs: {key}')
    return value


def parameter_count(model):
    """Count object identity once, regardless of fc3/head or base aliases."""
    return sum(p.numel() for p in {id(p): p for p in model.parameters()}.values())


def inspect_client(models, client_id, config, baseline_parameters):
    """Inspect actual in-memory local models without forward, recovery or RNG use."""
    from utils.multi_group_svd import model_capacity
    if not 0 <= client_id < config['num_clients'] or len(models) != config['num_server_models']:
        raise ValueError('Budget audit requires this batch N/K; recalculate for a different setup')
    identities = [set(map(id,model.parameters())) for model in models]
    if sum(map(len,identities)) != len(set().union(*identities)):
        raise ValueError('K local models must be independent Parameter objects, not repeated aliases')
    level = client_id % len(config['rank_rates'])
    rate = config['rank_rates'][level]
    rows = []
    for k, model in enumerate(models):
        if model.ratio_LR != rate:
            raise ValueError('Rank rate does not match this budget batch')
        kw = config['model_kwargs']
        if (model.conv1.in_channels != kw['in_features'] or model.conv1.out_channels != kw['n_kernels']
                or model.fc3.out_features != kw['num_classes']):
            raise ValueError('Model width/classes differ: recalculate parameter budgets')
        spatial = (((kw['input_size']-4)//2)-4)//2
        if model.fc_input_dim != 2*kw['n_kernels']*spatial*spatial:
            raise ValueError('Model input size differs: recalculate parameter budgets')
        count = parameter_count(model)
        if count != config['expected_single_parameters'][level]:
            raise ValueError('Instantiated local parameter count differs from the budget table')
        layers = model_capacity(model, {})['layers']
        rows.append(dict(model_id=k, parameter_count=count,
                         trainable_parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),
                         layers=layers))
    total = sum(r['parameter_count'] for r in rows)
    if total != config['expected_total_parameters'][level] or baseline_parameters != config['expected_baseline_parameters'][level]:
        raise ValueError('Actual baseline or K-model total differs: recalculate this budget table')
    if total > baseline_parameters:
        raise ValueError(f'Client {client_id}: K-model total exceeds baseline parameter budget')
    return dict(client_id=client_id, capacity_level=level, baseline_class=config['baseline_classes'][level],
        rank_rate=rate, baseline_parameters=baseline_parameters, local_models=rows,
        local_parameter_total=total, total_to_baseline_ratio=total/baseline_parameters,
        assigned_inference_parameter_count=rows[0]['parameter_count'],
        assigned_inference_rule='one ordinary post-local low-rank model selected by fixed membership',
        within_budget=True)


def build_audit(config_path=DEFAULT_CONFIG, capacity_manifest=None, group_file=None, previous_group_file=None):
    """Instantiate the repository's exact ordinary baselines and low-rank models.

    Optionally cross-check the *actual SVD download* capacity_manifest, not a
    reconstructed config string. All construction is enclosed by RNG restoration.
    """
    state = rng_state()
    try:
        from flcore.trainmodel import models as repository_models
        from utils.multi_group_svd import DEFAULT_RATIOS
        config = read_config(config_path)
        kw = config['model_kwargs']
        baseline_counts = []
        for name in config['baseline_classes']:
            if name not in [f'CNN_{i}_512' for i in range(1,6)]:
                raise ValueError('Expected original CNN-5-512 baseline classes')
            baseline = getattr(repository_models,name)(in_channels=kw['in_features'], n_kernels=kw['n_kernels'],out_dim=kw['num_classes'])
            baseline_counts.append(parameter_count(baseline))
        if baseline_counts != config['expected_baseline_parameters']:
            raise ValueError('Instantiated legacy baseline counts differ from configuration')
        full = repository_models.Hyper_CNN_512(**kw, ratio_LR=1.)
        full_count = parameter_count(full)
        clients = []
        for cid in range(config['num_clients']):
            rate = config['rank_rates'][cid % len(config['rank_rates'])]
            # Count K real instances. No forward, SVD or recovered upload is needed.
            local = [repository_models.Hyper_CNN_512(**kw,ratio_LR=rate) for _ in range(config['num_server_models'])]
            clients.append(inspect_client(local,cid,config,baseline_counts[cid % len(baseline_counts)]))
        old_single = [parameter_count(repository_models.Hyper_CNN_512(**kw,ratio_LR=r)) for r in DEFAULT_RATIOS]
        report = dict(schema_version=1, status='PASS', configuration=config,
            config_path=str(Path(config_path).resolve()), config_sha256=sha256(config_path),
            model_source='flcore.trainmodel.models', model_source_sha256=sha256(repository_models.__file__),
            counting_rule='unique Parameter identity, includes U/V, undecomposed layers, all biases and fc3; excludes buffers',
            clients=clients, baseline_parameters=baseline_counts, all_clients_within_budget=True,
            capacity_manifest_checked=False,
            server_full_parameters_per_model=full_count,
            server_full_parameter_total=config['num_server_models']*full_count,
            recovered_upload_cache_parameters_per_client=config['num_server_models']*full_count,
            logical_full_download_parameters_per_client_round=config['num_server_models']*full_count,
            logical_recovered_full_upload_parameters_per_client_round=config['num_server_models']*full_count,
            communication_budget_aligned=False, peak_memory_measured=False,
            communication_note='Unchanged full mother download / recovered full upload. Logical unique parameter elements, not measured network bytes, buffers, serialization overhead or peak memory.',
            local_data_epochs_per_client_round=config['num_server_models']*config['local_epochs_per_model'],
            previous_default_K1_reference=dict(rank_rates=list(DEFAULT_RATIOS),single_model_parameters=old_single,
                equal_budget_to_new_K3=False),
            grouping_note='New capacities generate new membership. Keep both old/new group lists; parameter count alone does not isolate membership changes.')
        if capacity_manifest is not None:
            manifest = json.loads(Path(capacity_manifest).read_text(encoding='utf-8'))
            protocol = manifest['protocol']
            n, rates = config['num_clients'], config['rank_rates']
            assignment = [i % len(rates) for i in range(n)]
            if (protocol['rank_rates'] != rates or protocol['client_level_ids'] != assignment or
                    protocol['client_rank_rates'] != [rates[i] for i in assignment] or
                    set(manifest['clients']) != {str(i) for i in range(n)}):
                raise ValueError('capacity_manifest belongs to a different capacity experiment')
            for client in clients:
                observed = manifest['clients'][str(client['client_id'])]
                if len(observed) != config['num_server_models']:
                    raise ValueError('capacity_manifest K mismatch')
                for actual, expected in zip(observed,client['local_models']):
                    if (actual['rank_rate'] != client['rank_rate'] or actual['layers'] != expected['layers'] or
                            actual['trainable_parameter_count'] != expected['parameter_count'] or
                            actual['recovered_parameter_count'] != full_count):
                        raise ValueError('Actual SVD download ranks/counts differ from budget audit')
            report.update(capacity_manifest_checked=True,capacity_manifest_sha256=sha256(capacity_manifest))
        for key, path in [('groups',group_file),('previous_groups',previous_group_file)]:
            if path is None:
                continue
            payload = json.loads(Path(path).read_text(encoding='utf-8'))
            from utils.multi_group_clustering import validate_group_file
            groups = validate_group_file(payload,payload['context'])
            if key == 'groups':
                protocol = payload['context'].get('svd_capacity_protocol',{})
                if (protocol.get('rank_rates') != config['rank_rates'] or payload['context']['K'] != config['num_server_models']
                        or protocol.get('client_level_ids') != [i % len(config['rank_rates']) for i in range(config['num_clients'])]
                        or payload['context']['num_clients'] != config['num_clients']
                        or payload['context']['model_family'] != config['model_family']
                        or [len(g) for g in groups] != config['group_sizes']):
                    raise ValueError('Grouping file is not from this new capacity batch')
                if capacity_manifest is not None and protocol != manifest['protocol']:
                    raise ValueError('Group and capacity manifests describe different protocols')
            group_protocol = payload['context'].get('svd_capacity_protocol',{})
            level_ids = group_protocol.get('client_level_ids')
            composition = None if level_ids is None else [
                {str(level):sum(level_ids[cid] == level for cid in group)
                 for level in range(len(group_protocol['rank_rates']))} for group in groups]
            report[key] = dict(path=str(Path(path).resolve()),sha256=sha256(path),members=groups,
                context=payload['context'],group_fingerprint=payload['group_fingerprint'],
                capacity_composition=composition)
        return report
    finally:
        set_rng(state)


def markdown_report(report):
    lines = ['# 三模型客户端参数预算审计', '',
        '实际实例化旧 CNN-5-512 五档基线及每客户端三个低秩模型；按 Parameter 身份去重，包含分类头与全部 bias。', '',
        '|档位|基线参数|秩比例|conv2/fc1/fc2秩|单模型参数|三模型参数|占预算|通过|',
        '|---|---:|---:|---|---:|---:|---:|---|']
    count = len(report['configuration']['rank_rates'])
    for c in report['clients'][:count]:
        m = c['local_models'][0]
        ranks = '/'.join(str(m['layers'][n]['rank']) for n in ('conv2','fc1','fc2'))
        lines.append(f"|{c['capacity_level']+1}|{c['baseline_parameters']:,}|{c['rank_rate']}|{ranks}|{m['parameter_count']:,}|{c['local_parameter_total']:,}|{100*c['total_to_baseline_ratio']:.6f}%|PASS|")
    lines += ['', '|客户端|档位|三模型参数|基线预算|通过|','|---|---:|---:|---:|---|']
    for c in report['clients']:
        lines.append(f"|{c['client_id']}|{c['capacity_level']+1}|{c['local_parameter_total']:,}|{c['baseline_parameters']:,}|PASS|")
    lines += ['', f"全部 {len(report['clients'])} 客户端通过。实际下载 capacity_manifest 交叉核验：{report['capacity_manifest_checked']}。",
        '', f"每个完整服务器/恢复上传模型：{report['server_full_parameters_per_model']:,} 参数；K模型合计 {report['server_full_parameter_total']:,}。",
        '客户端推理只使用所属的一个低秩模型；三模型存储总参数不等于单次推理参数量。',
        '当前仍传递完整母模型/恢复完整上传；没有对齐通信预算，没有测量峰值内存。上述逻辑传递计数不包括buffers及序列化开销。',
        f"每客户端每轮遍历数据 {report['local_data_epochs_per_client_round']} 次（每模型{report['configuration']['local_epochs_per_model']} epoch）；正式命令gr100仍为101轮。",
        '', '原默认K1低秩参考的实际单模型参数：'+', '.join(f'{v:,}' for v in report['previous_default_K1_reference']['single_model_parameters'])+'；与新K3不严格等参数。',
        '', '配置指纹：`'+report['config_sha256']+'`。改变K、宽度、输入或类别数须重新核算，不自动求秩。']
    for key in ('groups','previous_groups'):
        if key in report:
            lines += ['', f"{key}: `{report[key]['path']}`", '', '成员：`'+str(report[key]['members'])+'`。',
                      '容量构成（0起档位）：`'+str(report[key]['capacity_composition'])+'`。']
    lines += ['', '新旧容量重新自动分组，比较时保留两批名单；不能把全部差异归因于参数量。合成验收不代表真实准确率达标。', '']
    return '\n'.join(lines)
