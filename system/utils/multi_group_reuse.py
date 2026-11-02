"""Explicit membership reuse; default group-file validation is NOT weakened."""
import copy

from common.experiment import write_json
from utils.multi_group_clustering import digest_json,validate_group_file
from utils.multi_group_protocol import read_group_file


def context_differences(source,current,prefix=''):
    result=[]
    for key in sorted(set(source)|set(current)):
        name=prefix+'.'+key if prefix else key
        if key in source and key in current and isinstance(source[key],dict) and isinstance(current[key],dict):
            result.extend(context_differences(source[key],current[key],name))
        elif source.get(key)!=current.get(key) or (key in source)!=(key in current):
            result.append(dict(field=name,source=source.get(key),current=current.get(key)))
    return result


def validate_members(payload,current):
    """Validate source integrity FIRST, then only permit named protocol changes."""
    if not isinstance(payload,dict) or not isinstance(payload.get('context'),dict):raise ValueError('Group file lacks source context')
    source=payload['context'];groups=validate_group_file(payload,source)
    # These fields bind data, IDs, N/K, architecture, common mother and actual
    # client rank assignment. Never exempt them, even with identical member lists.
    for key in ('num_clients','K','model_family','capacities','initial_parameter_sha256','data_fingerprint',
                'svd_capacity_protocol','svd_initialization'):
        if key not in source or source[key]!=current.get(key):
            raise ValueError(f'Controlled membership reuse: immutable {key} mismatch/missing; do not reuse this grouping')
    differences=context_differences(source,current)
    allowed=('training.grad_clip','sampling_protocol','grouping_protocol')
    for row in differences:
        if not any(row['field']==key or row['field'].startswith(key+'.') for key in allowed):
            raise ValueError('Controlled membership reuse refuses undeclared context difference: '+row['field'])
    return groups,differences


def load_members(server):
    args=server.args
    if not args.multi_group_file:raise ValueError('Last-epoch protocol requires an existing group file; automatic clustering is disabled')
    controlled=getattr(args,'multi_group_reuse_policy','strict')=='members_only'
    def validate(payload):
        return validate_members(payload,server.context)[0] if controlled else validate_group_file(payload,server.context)
    payload,sha,waited=read_group_file(args.multi_group_file,validate=validate)  # Missing fails immediately; no re-training/wait.
    groups=validate(payload)
    differences=validate_members(payload,server.context)[1] if controlled else []
    receipt=dict(policy=args.multi_group_reuse_policy,source_path=str(args.multi_group_file),source_file_sha256=sha,
        source_context=copy.deepcopy(payload['context']),current_context=copy.deepcopy(server.context),
        protocol_differences=differences,source_group_fingerprint=payload['group_fingerprint'],
        immutable_data_model_initialization_capacity_checked=True,source_integrity_checked=True,
        membership_reused_unchanged=True,reclustered=False,waited_seconds=waited)
    # Keep the ORIGINAL signed artifact as provenance, never pretend it was
    # generated with this run's different sampling/clip/formation protocol.
    server.groups=groups;server.group_payload=payload;server.group_source_sha256=sha
    server.group_reuse_receipt=receipt
    from pathlib import Path
    write_json(Path(server.report_dir())/'group_reuse_verification.json',receipt)
    write_json(Path(server.report_dir())/'fixed_groups.json',payload)
    return receipt
