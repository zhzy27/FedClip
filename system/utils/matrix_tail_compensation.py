"""Sample-mass subgroup MATRIX additions to an unchanged full Avg base.

All subgroup matrices are original recovered uploads. Residual SVD supplies
its OWN singular values/vectors. No factor averaging or all_tail scaling here.
"""
import math
import time

import torch

LAYERS=('conv2','fc1','fc2')


def _checked(matrix,shape=None):
    if not isinstance(matrix,torch.Tensor) or matrix.ndim!=2 or not matrix.is_floating_point():raise ValueError('Need a floating parameter matrix')
    if shape is not None and tuple(matrix.shape)!=tuple(shape):raise ValueError('Matrix shape changed')
    if not torch.isfinite(matrix).all():raise FloatingPointError('Nonfinite matrix compensation input')
    return matrix.detach().to(device='cpu',dtype=torch.float64)


def prefix_overlap(before,after,rank):
    value=float((before[:,:rank].T@after[:,:rank]).square().sum()/rank)
    return min(1.,max(0.,value))


@torch.no_grad()
def compensate_matrix(average,read_matrix,ranks,samples,order,*,rank_rtol=None,rank_atol=0.):
    """Return same dtype/device matrix plus bounded diagnostics, without writes.

    Relative rank tolerance defaults to max(m,n)*max(input_eps,float64_eps),
    scaled by the unprojected subgroup's exact spectral norm. This accounts for
    float32 upload precision and cancellation even though arithmetic is float64.
    """
    if (not callable(read_matrix) or not samples or set(ranks)!=set(samples) or set(order)!=set(samples)
        or len(order)!=len(set(order)) or any(type(n) is not int or n<=0 for n in samples.values())):
        raise ValueError('Need complete unique ranks/order and original positive train counts')
    source=_checked(average);m,n=source.shape
    if any(type(r) is not int or not 1<=r<=min(m,n) for r in ranks.values()):raise ValueError('Illegal actual layer rank')
    if not math.isfinite(rank_atol) or rank_atol<0 or rank_rtol is not None and (not math.isfinite(rank_rtol) or rank_rtol<0):
        raise ValueError('Rank tolerances must be finite nonnegative')
    epsilon=max(torch.finfo(average.dtype).eps,torch.finfo(torch.float64).eps)
    relative=max(m,n)*epsilon if rank_rtol is None else rank_rtol
    tick=time.perf_counter();left,s_before,vh_before=torch.linalg.svd(source,full_matrices=False)
    tiers=sorted(set(ranks.values()));minimum=tiers[0];up=left[:,:minimum];vp=vh_before[:minimum].T
    addition=torch.zeros_like(source);segments=[];mass=sum(samples.values())
    for low,high in zip(tiers,tiers[1:]):
        started=time.perf_counter();owners=[cid for cid in order if ranks[cid]>=high]
        subtotal=sum(samples[cid] for cid in owners);q=subtotal/mass
        subgroup=torch.zeros_like(source)
        for cid in owners:subgroup.add_(_checked(read_matrix(cid),source.shape)*(samples[cid]/subtotal))
        # B is the FULL original-upload subgroup mean, never corrected Avg.
        b_s=torch.linalg.svdvals(subgroup);scale=float(b_s[0]) if len(b_s) else 0.
        before=float(subgroup.square().sum())
        residual=subgroup-up@(up.T@subgroup);after_left=float(residual.square().sum())
        residual=residual-(residual@vp)@vp.T
        after_both=float(residual.square().sum());u,s,vh=torch.linalg.svd(residual,full_matrices=False)
        threshold=max(rank_atol,relative*scale);numerical=int((s>threshold).sum());planned=high-low;take=min(planned,numerical)
        prefix_before=up.shape[1]
        if take:
            term=(u[:,:take]*s[:take])@vh[:take]
            addition.add_((1-q)*term)
            # QR changes only the projector basis, not T or its singular values.
            # Its span is exactly the actually selected past directions.
            up,ru=torch.linalg.qr(torch.cat([up,u[:,:take]],dim=1),mode='reduced')
            vp,rv=torch.linalg.qr(torch.cat([vp,vh[:take].T],dim=1),mode='reduced')
            qr_floor=max(m,n)*torch.finfo(torch.float64).eps
            if (ru.diag().abs().min()<qr_floor or rv.diag().abs().min()<qr_floor):raise FloatingPointError('Selected directions numerically repeat the projection prefix')
            term_norm=float(term.norm());added_norm=(1-q)*term_norm
        else:term_norm=added_norm=0.
        segments.append(dict(partition_start=low+1,partition_end=high,contributor_ids=owners,
            contributor_train_samples=[samples[i] for i in owners],contributor_train_samples_total=subtotal,
            all_train_samples=mass,q=q,compensation_coefficient=1-q,
            subgroup_weights=[samples[i]/subtotal for i in owners],subgroup_weight_sum=math.fsum(samples[i]/subtotal for i in owners),
            energy_before_projection=before,energy_after_left_projection=after_left,energy_after_both_projections=after_both,
            planned_rank=planned,numerical_rank=numerical,actual_taken_rank=take,rank_shortfall=planned-take,
            rank_threshold=threshold,rank_rtol=relative,rank_atol=rank_atol,rank_scale=scale,
            rank_scale_definition='spectral norm of unprojected original subgroup mean B',
            input_epsilon=epsilon,computation_dtype='float64 CPU',prefix_rank_before=prefix_before,prefix_rank_after=up.shape[1],
            residual_top_sigma=float(s[0]) if len(s) else 0.,last_selected_sigma=float(s[take-1]) if take else None,
            selected_singular_values=s[:take].tolist(),selected_component_energy=float(s[:take].square().sum()),
            T_frobenius_norm=term_norm,added_matrix_frobenius_norm=added_norm,
            prefix_left_orthogonality_error=float((up.T@up-torch.eye(up.shape[1],dtype=up.dtype)).norm()),
            prefix_right_orthogonality_error=float((vp.T@vp-torch.eye(vp.shape[1],dtype=vp.dtype)).norm()),
            segment_seconds=time.perf_counter()-started))
    corrected=(source+addition).to(average)
    _checked(corrected,source.shape)
    # Compare actual dtype-rounded complete matrices, including all reordering.
    u_after,s_after,vh_after=torch.linalg.svd(corrected.detach().cpu().double(),full_matrices=False)
    boundaries=[]
    for rank in tiers:
        before_at=float(s_before[rank-1]);after_at=float(s_after[rank-1]);has_next=rank<len(s_before)
        boundaries.append(dict(boundary_rank=rank,before_sigma_at=before_at,after_sigma_at=after_at,
            before_sigma_next=float(s_before[rank]) if has_next else None,after_sigma_next=float(s_after[rank]) if has_next else None,
            before_next_over_at=float(s_before[rank]/s_before[rank-1]) if has_next and before_at else None,
            after_next_over_at=float(s_after[rank]/s_after[rank-1]) if has_next and after_at else None,
            prefix_left_subspace_overlap=prefix_overlap(left,u_after,rank),
            prefix_right_subspace_overlap=prefix_overlap(vh_before.T,vh_after.T,rank),
            overlap_definition='||U_before[:r].T U_after[:r]||_F^2 / r, invariant to signs and rotations within subspace',
            before_boundary_nearly_degenerate=bool(has_next and abs(s_before[rank-1]-s_before[rank])<=relative*float(s_before[0])),
            after_boundary_nearly_degenerate=bool(has_next and abs(s_after[rank-1]-s_after[rank])<=relative*float(s_after[0]))))
    return corrected,dict(segments=segments,boundaries=boundaries,
        singular_before=s_before.tolist(),singular_after=s_after.tolist(),
        original_Avg_frobenius_norm=float(source.norm()),added_total_frobenius_norm=float(addition.norm()),
        actual_dtype_matrix_change_frobenius=float((corrected.detach().cpu().double()-source).norm()),
        total_seconds=time.perf_counter()-tick,actual_rank_tiers=tiers,
        prefix_basis='minimum-rank W_avg SVD, then QR spans of ACTUAL selected residual-SVD directions; no deficit carry-forward',
        base_rule='original W_avg plus all (1-q)*T; subgroup B always from original full uploads')


def weight_matrix(weight,layer):
    if isinstance(layer,torch.nn.Conv2d):
        k=layer.kernel_size[0]
        return weight.permute(0,2,1,3).reshape(layer.out_channels*k,layer.in_channels*k)
    return weight


def matrix_weight(matrix,layer):
    if isinstance(layer,torch.nn.Conv2d):
        k=layer.kernel_size[0]
        return matrix.reshape(layer.out_channels,k,layer.in_channels,k).permute(0,2,1,3)
    return matrix


@torch.no_grad()
def compensate_model(average,files,ranks_by_layer,samples,order,*,rank_rtol=None,rank_atol=0.):
    import copy
    from utils.multi_rank_partition import actual_matrix,protected_rng
    result=copy.deepcopy(average);records=[]
    canonical={id(p):name for name,p in average.named_parameters()}
    with protected_rng([average]):
        for name in LAYERS:
            layer=getattr(average,name);key=canonical[id(layer.weight)]
            def read(cid):
                values=torch.load(files[cid],map_location='cpu',weights_only=False) if isinstance(files[cid],str) or hasattr(files[cid],'read_bytes') else files[cid]
                return weight_matrix(values[key],layer)
            corrected,report=compensate_matrix(actual_matrix(layer),read,ranks_by_layer[name],samples,order,
                rank_rtol=rank_rtol,rank_atol=rank_atol)
            getattr(result,name).weight.copy_(matrix_weight(corrected,layer))
            records.append(dict(layer=name,canonical_parameter=key,**report))
    return result,records
