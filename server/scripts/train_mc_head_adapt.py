#!/usr/bin/env python3
"""Freeze frontend+lstm; adapt attn/head on native multicrop crops."""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.normalize import FEATURE_DIM
from pipeline.sequence_model import build_sequence_model, load_ship_checkpoint
from scripts.eval_multicrop import load_wlasl100
from scripts.train_multicrop_matched_kd import NativeCropDataset, multicrop_logits, val_ab_indices
from scripts.train_ondevice_coreml import topk_acc, nmm_aux_targets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init-from', type=Path, required=True)
    ap.add_argument('--teacher', type=Path, default=ROOT/'models'/'sign_classifier.pt')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=36)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--seed', type=int, default=71)
    ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--ship-test', type=float, default=0.5968992114067078)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    blob = np.load(ROOT/'models'/'pose_features_dense.npz', allow_pickle=True)
    labels = [str(x) for x in blob['labels'].tolist()]
    cache = torch.load(ROOT/'models'/'native_train_cache_mc.pt', map_location='cpu', weights_only=False)
    xs, ys, srcs = cache['xs'], cache['ys'], cache['sources']
    aux = nmm_aux_targets(labels, ys)
    src_arr = np.array(srcs)
    counts = np.bincount(ys, minlength=len(labels)).astype(np.float64).clip(1, None)
    class_w = 1.0/np.sqrt(counts); class_w = class_w/class_w.sum()*len(labels)
    sample_w = class_w[ys].astype(np.float64).copy()
    sample_w[src_arr=='wlasl100'] *= 3.5
    sample_w[src_arr=='wlasl300'] *= 1.2
    sample_w[np.char.startswith(src_arr,'aslcitizen')] *= 1.15
    sample_w[src_arr=='synth'] *= 0.35
    va_x, va_y = load_wlasl100('Val', labels); te_x, te_y = load_wlasl100('Test', labels)
    a_idx, b_idx = val_ab_indices(len(va_y), 0)
    teacher, _, _ = load_ship_checkpoint(args.teacher); teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    init = torch.load(args.init_from, map_location='cpu', weights_only=False)
    model = build_sequence_model('tcn-bilstm', input_dim=FEATURE_DIM, hidden_dim=int(init.get('hidden_dim',192)),
        num_layers=int(init.get('num_layers',2)), num_classes=len(labels), bidirectional=True, dropout=0.20)
    model.load_state_dict(init['state_dict'], strict=False)
    for name, p in model.named_parameters():
        if name.startswith('frontend') or name.startswith('lstm'):
            p.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print('trainable', sum(p.numel() for p in trainable))
    ds = NativeCropDataset(xs, ys, aux, srcs, 32, 96, args.seed, True, 2)
    loader = DataLoader(ds, batch_size=args.batch,
        sampler=WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double), len(sample_w), True),
        num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=torch.as_tensor(class_w, dtype=torch.float32), label_smoothing=0.02)
    bce = nn.BCEWithLogitsLoss(); kl = nn.KLDivLoss(reduction='batchmean')
    def score_mc(m):
        Lva = multicrop_logits(m, va_x, 5, 96); Lte = multicrop_logits(m, te_x, 5, 96)
        yva = torch.as_tensor(va_y); yte = torch.as_tensor(te_y)
        sa = float((Lva[a_idx].argmax(-1)==yva[a_idx]).float().mean())
        sb = float((Lva[b_idx].argmax(-1)==yva[b_idx]).float().mean())
        return dict(val_A=sa, val_B=sb, val_top1=float(topk_acc(Lva,yva,1)), test_top1=float(topk_acc(Lte,yte,1)),
                    test_top5=float(topk_acc(Lte,yte,5)), correct=int((Lte.argmax(-1)==yte).sum()), n_test=len(te_y))
    model.eval(); base = score_mc(model)
    print(f'baseline A={base["val_A"]:.3f} B={base["val_B"]:.3f} val={base["val_top1"]:.3f} te={base["test_top1"]:.3f} ({base["correct"]}/258)')
    best_sum = base['val_A']+base['val_B']; best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; best=base
    for epoch in range(1, args.epochs+1):
        progress = (epoch-1)/max(args.epochs-1,1)
        lr_now = args.lr*0.2 + 0.5*(args.lr-args.lr*0.2)*(1+math.cos(math.pi*progress))
        for pg in opt.param_groups: pg['lr']=lr_now
        model.train(); ds.epoch_salt=epoch; tot=0; n=0; t0=time.time()
        for xb,yb,ab in loader:
            B,C,T,D = xb.shape
            xb = xb.reshape(B*C,T,D); yb = yb.long().repeat_interleave(C); ab = ab.float().repeat_interleave(C,0)
            opt.zero_grad()
            with torch.no_grad(): tlog = teacher(xb)
            logits, aux_l = model(xb, return_aux=True)
            loss = crit(logits,yb)+0.2*bce(aux_l,ab)
            Td=2.5
            loss = 0.85*kl(torch.log_softmax(logits/Td,-1), torch.softmax(tlog/Td,-1))*(Td*Td) + 0.15*loss
            loss.backward(); nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
            tot += float(loss)*B; n += B
        model.eval(); m = score_mc(model)
        print(f'epoch {epoch:02d} loss={tot/max(n,1):.4f} A={m["val_A"]:.3f} B={m["val_B"]:.3f} val={m["val_top1"]:.3f} te={m["test_top1"]:.3f} ({m["correct"]}/258) lr={lr_now:.2e} {time.time()-t0:.1f}s')
        if m['val_A']+m['val_B'] >= best_sum-1e-9 and m['val_A']+0.012>=best['val_A'] and m['val_B']+0.012>=best['val_B']:
            best_sum = m['val_A']+m['val_B']; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; best=m
            print('  * best')
    model.load_state_dict(best_state); final=score_mc(model)
    print('FINAL', final)
    ckpt = {**{k:init[k] for k in init if k!='state_dict'}, 'state_dict':best_state,
            'note': f'mc head-only adapt from {args.init_from.name}',
            'wlasl100_subset': {**final, 'method':'mc head-only', 'multi_crop_count':5, 'max_context_frames':96},
            'comparable_wlasl100': {'val_top1':final['val_top1'],'test_top1':final['test_top1'],'test_top5':final['test_top5'],'n_val':len(va_y),'n_test':258,'method':'multicrop nc=5 max96 plain'}}
    torch.save(ckpt, args.out)
    args.report.write_text(json.dumps({'arch':'mc-head-only','init':str(args.init_from),'wlasl100_subset':ckpt['wlasl100_subset'],'comparable_wlasl100':ckpt['comparable_wlasl100'],'delta_pp':(final['test_top1']-args.ship_test)*100}, indent=2))
    print('saved', args.out)

if __name__ == '__main__':
    raise SystemExit(main())
