"""Pure fc1 SVD gauge shared by offline B and opt-in client downloads."""
import math


def scales(singular,minimum,rank,strength=1.):
    import torch
    if not math.isfinite(strength) or strength<0:raise ValueError('Tail scale strength must be finite nonnegative')
    if (singular.ndim!=1 or not 0<minimum<rank<=len(singular) or not singular.is_floating_point()
        or not torch.isfinite(singular).all() or (singular<0).any()):raise ValueError('Invalid same-SVD singular values or fc1 rank range')
    floor=singular[minimum-1].sqrt()*strength
    if not torch.isfinite(floor):raise FloatingPointError('Nonfinite configured factor floor')
    a=torch.maximum(singular[minimum:rank].sqrt(),floor)
    b=torch.where(a>0,singular[minimum:rank]/a.clamp_min(torch.finfo(a.dtype).tiny),torch.zeros_like(a))
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise FloatingPointError('Nonfinite tail factor scales')
    return a,b,floor


def apply_from_svd(layer,left,singular,right_h,minimum,strength=1.,verify_product=False):
    """No new decomposition. Values move to the ACTUAL factor device/dtype.

    In legacy Decom_LINEAR these new factors are CPU tensors even if the mother
    SVD runs on CUDA. This matches offline B's arithmetic on captured CPU P/S/Qh.
    """
    import torch
    with torch.no_grad():
        u,v=layer.weight_u,layer.weight_v;rank=layer.rank
        s=singular.detach().to(u);p=left[:,minimum:rank].detach().to(u);qh=right_h[minimum:rank,:].detach().to(v)
        if (p.shape!=(u.shape[0],rank-minimum) or qh.shape!=(rank-minimum,v.shape[1])
            or not torch.isfinite(p).all() or not torch.isfinite(qh).all()):raise ValueError('Wrong/nonfinite same-SVD unit vectors')
        before=(u.detach().cpu().double()@v.detach().cpu().double()) if verify_product else None
        a,b,floor=scales(s,minimum,rank,strength)
        u[:,minimum:].copy_(p*a);v[minimum:,:].copy_(b[:,None]*qh)
        if not torch.isfinite(u).all() or not torch.isfinite(v).all():raise FloatingPointError('Nonfinite adjusted fc1 factors')
        result=dict(floor_singular_value=float(s[minimum-1]),floor_factor_scale=float(floor),
            floor_active_tail_slots=int((floor>s[minimum:rank].sqrt()).sum()),
            zero_tail_singular_values=int((s[minimum:rank]==0).sum()),zero_floor=bool(floor==0),
            zero_singular_rule='use same-SVD unit vectors; zero sigma/positive floor gives nonzero U and zero V; zero floor and sigma give both zero')
        if before is not None:
            after=u.detach().cpu().double()@v.detach().cpu().double();error=after-before
            if not torch.allclose(before,after,atol=1e-6,rtol=1e-5):raise ValueError('Tail gauge failed restored fc1 product equivalence')
            result.update(restored_fc1_max_absolute_error=float(error.abs().max()),
                restored_fc1_change_frobenius=float(error.norm()),product_check_atol=1e-6,product_check_rtol=1e-5,
                product_check_dtype='float64 CPU products of actual before/after factor tensors')
        return result
