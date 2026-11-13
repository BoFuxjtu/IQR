"""Environmental heterogeneity and held-out prediction (standalone experiment).

python iqr_environment_study.py --mode full --workers 4    # refit, 200 repetitions
python iqr_environment_study.py --mode report --input results    # reuse fits
python iqr_environment_study.py --mode quick              # small example
python iqr_environment_study.py --mode fresh --input results     # new test data only
"""
import os
# Parallelize replicates, not the tiny regressions inside them.
for var in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[var] = '1'
import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog, OptimizeWarning
from scipy.special import ndtr

SCENARIOS = {'strong': (.2, .8), 'weak': (.45, .55), 'identical': (.5, .5)}
TEST_A = (0., -.8)
NAMES = ['Z', 'W', 'D']
RULE = 'A fitted slope is selected if and only if its coefficient is nonzero; intercept excluded.'
BITS = 1 << np.arange(3)
CELLS = np.column_stack((np.ones(8), 2. * ((np.arange(8)[:, None] & BITS) != 0) - 1.))
OPTIONS = dict(maxiter=5000, maxfun=50000, ftol=1e-12, gtol=1e-10)


# Data generation. Reuse the same underlying draws across training contrasts.
def draw_data(seed, rep, split, e, n):
    def rng(k):
        return np.random.default_rng([seed, rep, split, e, k])
    z = 2. * rng(0).integers(0, 2, n) - 1.
    w = 2. * rng(1).integers(0, 2, n) - 1.
    u = rng(2).standard_normal(n)
    return z, w, u, rng(3).random(n)


def environment(draws, a):
    z, w, u, v = draws
    d = np.where(v < (1. + a * np.tanh(u)) / 2., 1., -1.)
    return np.column_stack((z, w, d)), 1. + 2.*z + (1. + w/2.)*u


# There are only eight possible rows (Z,W,D). Summing scores by row makes
# the gradient fast without approximating either the loss or the penalty.
def fit_sqr(y, obs, group, n, mask, h, tau, gamma):
    cols = [0] + [j+1 for j in range(3) if mask & (1 << j)]
    X = np.ascontiguousarray(CELLS[:, cols])
    def cell_sum(v):
        return np.bincount(group, weights=v, minlength=16).reshape(2, 8)
    def objective(b):
        r = y - (X @ b)[obs]
        v = r / h
        f = (2.*np.pi)**(-.5) * np.exp(-.5*v*v)
        psi = ndtr(-v) - tau
        loss = np.mean(-r*psi + h*f)
        G = (cell_sum(psi) * (1./n)) @ X
        grad = G.mean(axis=0)
        if gamma:
            diag = cell_sum(f) * ((1./n)/h)
            grad += 2.*gamma*((G @ X.T * diag).sum(axis=0) @ X)/2
            loss += gamma*np.einsum('ij,ij->', G, G)/2
        return loss, grad
    res = minimize(objective, np.zeros(len(cols)), jac=True, method='L-BFGS-B', options=OPTIONS)
    b = np.zeros(4)
    b[cols] = res.x
    if not res.success or not np.isfinite(res.fun):
        return np.full(4, np.nan), np.inf, mask
    return b, res.fun, mask


def fit_eills(y, obs, group, n, gamma):
    counts = np.bincount(group, minlength=16).reshape(2, 8) * (1./n)
    sums = np.bincount(group, weights=y, minlength=16).reshape(2, 8) * (1./n)
    scores = []
    for mask in range(8):
        cols = [0] + [j+1 for j in range(3) if mask & (1 << j)]
        X = np.ascontiguousarray(CELLS[:, cols])
        M = np.einsum('ci,ec,cj->eij', X, counts, X)
        t = sums @ X
        A = (M + gamma*(M @ M)).mean(axis=0)
        rhs = (t + gamma*(M @ t[..., None])[..., 0]).mean(axis=0)
        try:
            b = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            b = np.linalg.lstsq(A, rhs, rcond=None)[0]
        r = y - (X @ b)[obs]; g = M @ b - t
        risk = .5*(np.mean(r*r) + gamma*np.sum(g*g)/2.)
        padded = np.zeros(4); padded[cols] = b
        scores.append((padded, risk, mask))
    return min(scores, key=lambda fit: fit[1])


def pooled_qr(y, X, tau):
    # Dual LP gives all four primal coefficients via equality multipliers.
    X = np.column_stack((np.ones(len(y)), X))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected:.*threads', category=OptimizeWarning)
        res = linprog(-y, A_eq=X.T, b_eq=np.zeros(4), bounds=((1./len(y))*(tau-1), (1./len(y))*tau),
                      method='highs-ds', options={'threads': 1})
    return (-res.eqlin.marginals if res.success else np.full(4, np.nan)), 0., 7


# Held-out check loss and signed coverage deviations in the two test environments.
# Each fit predicts a constant within each of the eight covariate cells.
def prepare_test(X, y):
    cells = []
    for x in CELLS:
        ys = np.sort(y[(X == x[1:]).all(axis=1)])
        if len(ys):
            cells.append((x, ys, np.r_[0., np.cumsum(ys)]))
    return cells


def score_test(cells, beta, tau):
    beta, tau = np.atleast_2d(beta), np.asarray(tau)
    loss = np.zeros(len(beta)); hits = np.zeros((2, len(beta))); sizes = np.zeros(2)
    for x, ys, cs in cells:
        q = beta @ x; k = np.searchsorted(ys, q, side='right')
        m = len(ys); j = int(x[2] > 0)
        loss += (1.-tau)*(k*q-cs[k]) + tau*(cs[-1]-cs[k]-(m-k)*q)
        hits[j] += k; sizes[j] += m
    return np.column_stack((loss/sizes.sum(), hits.sum(axis=0)/sizes.sum()-tau,
                            hits[0]/sizes[0]-tau if sizes[0] else np.full(len(beta), np.nan),
                            hits[1]/sizes[1]-tau if sizes[1] else np.full(len(beta), np.nan)))


def one_rep(job):
    n, rep, directory, cfg = job
    gamma, seed = cfg['gamma'], cfg['seed']
    train = [draw_data(seed, rep, 0, e, n) for e in range(2)]
    test = [prepare_test(*environment(draw_data(seed, rep, 1, e, cfg['test_n']), a))
            for e, a in enumerate(TEST_A)]
    records = []
    for scenario, shifts in SCENARIOS.items():
        xys = [environment(draw, a) for draw, a in zip(train, shifts)]
        X = np.vstack([xy[0] for xy in xys]); y = np.concatenate([xy[1] for xy in xys])
        obs = (X > 0) @ BITS; env = np.repeat([0, 1], n); group = 8*env + obs
        if not cfg['primary_only']:
            eills = fit_eills(y, obs, group, n, gamma)
        for tau in (.5, .9):
            h = cfg['h'] or np.sqrt(3*tau*(1-tau)*(gamma if gamma > 0 else 20.)/(2*n))
            q = NormalDist().inv_cdf(tau); truth = np.array([1.+q, 2., q/2., 0.])
            def fit(mask, penalty):
                return fit_sqr(y, obs, group, n, mask, h, tau, penalty)
            pooled = [fit(k, 0.) for k in (range(8) if not cfg['primary_only'] else [7])]
            ks = [fit(k, gamma) for k in range(8)] if gamma else [pooled[-1]]
            fits = [('KS-IQR', min(ks, key=lambda v:v[1]), sum(np.isinf(v[1]) for v in ks), gamma, h),
                    ('Pooled-SQR', pooled[-1], int(np.isinf(pooled[-1][1])), 0., h)]
            if not cfg['primary_only']:
                def bic(v):
                    return 4*n*np.log(v[1]) + bin(v[2]).count('1')*np.log(2*n)
                qr = pooled_qr(y, X, tau)
                fits.extend([('Sparse-pooled-SQR', min(pooled, key=bic), sum(np.isinf(v[1]) for v in pooled), 0., h),
                             ('Pooled-QR', qr, int(not np.isfinite(qr[0]).all()), 0., None),
                             ('EILLS', eills, 0, gamma, None)])
            fits.append(('Structural-oracle', (truth, 0., 1 + 2*int(q != 0)), 0, None, None))
            for method, (b, _, mask), failures, fg, fh in fits:
                ok = np.isfinite(b).all()
                row = dict(n=n, rep=rep, scenario=scenario, tau=tau, gamma=gamma, h=h, method=method,
                           success=int(ok), failed_mask_count=failures,
                           mask=';'.join(name for j,name in enumerate(NAMES) if mask & (1<<j)) if ok else None,
                           fitting_gamma=fg, fitting_h=fh)
                row.update(zip(['B0','BZ','BW','BD'], b))
                # Excluded slopes are exactly zero; the intercept is never counted.
                row.update({'sel_'+name:int(v != 0) if ok else None for name,v in zip(NAMES,b[1:])})
                eligible = ok and (method != 'EILLS' or tau == .5)
                row['coef_err'] = np.linalg.norm(b-truth) if eligible else None
                for e,cells in enumerate(test):
                    values = score_test(cells,b,tau)[0] if eligible else [np.nan]*4
                    row.update(zip([f'test{e}_{s}' for s in ('loss','cov','cov_wminus','cov_wplus')], values))
                losses = [row['test0_loss'], row['test1_loss']]
                row.update(worst_pinball=max(losses), mean_pinball=np.mean(losses),
                           coverage_abs_worst=max(abs(row['test0_cov']), abs(row['test1_cov'])))
                records.append(row)
    dest = directory/'parts'/f'n{n:06d}_r{rep:05d}.csv'
    tmp = dest.with_suffix('.tmp')
    pd.DataFrame(records).to_csv(tmp, index=False)
    tmp.replace(dest)
# Reports use the actual fitted slopes, including when reading results of an older run.
GROUP = ['scenario', 'n', 'tau', 'gamma', 'h']
SLOPES = ['Z', 'W', 'D']
SEL = ['sel_' + s for s in SLOPES]
TESTCOL = [f'test{e}_{s}' for e in (0,1) for s in ('loss','cov','cov_wminus','cov_wplus')]

def set_selection(raw):
    ok = raw.success.eq(1)
    for s in SLOPES:
        b = raw['B' + s]
        raw['sel_' + s] = (b != 0).astype(float).where(ok & b.notna())
    return raw

def read_saved(source):
    import zipfile
    path = Path(source)
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if Path(n).name == 'replicates.csv')
            with z.open(name) as f:
                raw = pd.read_csv(f, float_precision='round_trip')
            meta = json.loads(z.read(Path(name).with_name('manifest.json').as_posix()))
    else:
        base = path if path.is_dir() else path.parent
        raw = pd.read_csv(base/'replicates.csv', float_precision='round_trip')
        meta = json.loads((base/'manifest.json').read_text(encoding='utf8'))
    return set_selection(raw), meta

def summarize(raw, keys, metrics):
    x = raw.melt(id_vars=keys, value_vars=metrics, var_name='metric')
    return x.groupby(keys+['metric']).value.agg(mean='mean',count='count',mcse='sem').reset_index()

def paired(raw, metrics):
    fits = raw.set_index(GROUP+['rep','method'])[metrics]
    ks = fits.xs('KS-IQR',level='method')
    out = []
    for name in raw.method.unique():
        if name != 'KS-IQR':
            dif = ks.sub(fits.xs(name,level='method')).reset_index()
            out.append(summarize(dif,GROUP,metrics).assign(reference=name))
    return pd.concat(out,ignore_index=True)

def save_fit_summary(raw, fit_dir):
    metrics = SEL+['coef_err','worst_pinball','mean_pinball','coverage_abs_worst']+TESTCOL
    keys = GROUP+['method']
    data = set_selection(raw).loc[raw.success.eq(1)]
    fit = raw.groupby(keys).agg(n_tasks=('rep','size'),n_success=('success','sum'),
                               failed_mask_count_sum=('failed_mask_count','sum'))
    fit['n_fit_failed'] = fit.n_tasks-fit.n_success
    summary = summarize(data,keys,metrics+['B0','BZ','BW','BD']).pivot(index=keys,columns='metric')
    summary.columns = [f"{m}_{'n' if k=='count' else k}" for k,m in summary.columns]
    fit.join(summary).to_csv(fit_dir/'summary.csv')
    tab = paired(data, metrics).rename(columns={'count':'n_pairs','mean':'mean_difference','mcse':'paired_mcse'})
    tab.assign(method='KS-IQR').to_csv(fit_dir/'paired_differences.csv',index=False)

def report(raw, directory, figure_n=None, plot=True):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    keys = GROUP+['method']
    counts = raw.groupby(keys).success.agg(attempted='size',successful='sum')
    counts.assign(failed=counts.attempted-counts.successful).to_csv(directory/'fit_counts.csv')
    data = set_selection(raw).loc[raw.success.eq(1)].copy()
    data.loc[(data.method=='EILLS') & (data.tau!=.5), TESTCOL] = np.nan
    data['worst_loss'] = data[['test0_loss','test1_loss']].max(axis=1, skipna=False)
    extra = []
    for e in (0,1):
        cov = data[f'test{e}_cov']
        data[f'test{e}_coverage'] = cov+data.tau
        data[f'test{e}_abs_cov'] = cov.abs()
        extra += [f'test{e}_coverage',f'test{e}_abs_cov']
    metrics = SEL+['worst_loss']+TESTCOL+extra
    summary = summarize(data,keys,metrics)
    summary.to_csv(directory/'heldout_summary.csv',index=False)
    dif = paired(data,metrics)
    dif.assign(contrast='KS-IQR minus '+dif.reference).drop(columns='reference').to_csv(directory/'heldout_paired.csv',index=False)
    env = summary[summary.metric.str.startswith('test')].copy()
    env['test_a'] = np.where(env.metric.str.startswith('test0_'),0.,-.8)
    env['metric'] = env.metric.str[6:]
    env.to_csv(directory/'by_test_environment.csv',index=False)
    if plot:
        sample_sizes = sorted(raw['n'].unique()) if figure_n is None else [figure_n]
        for n in sample_sizes:
            plot_study(summary, n, directory)
    return summary

def plot_study(summary, n, directory):
    import matplotlib.pyplot as plt

    # EILLS and KS-IQR use the color/marker/line combinations of Figure 3.
    # Pooled SQR uses two additional colors and markers from the same palette.
    styles = [
        ('EILLS',      .5, 'EILLS',                       '#E41A1C', 'D', '-'),
        ('KS-IQR',     .5, r'KS-IQR ($\tau=0.5$)',        '#377EB8', 'o', '--'),
        ('KS-IQR',     .9, r'KS-IQR ($\tau=0.9$)',        '#A65628', '<', '--'),
        ('Pooled-SQR', .5, r'Pooled SQR ($\tau=0.5$)',    '#4DAF4A', 'P', ':'),
        ('Pooled-SQR', .9, r'Pooled SQR ($\tau=0.9$)',    '#FF7F00', 's', '-.'),
    ]
    scenarios = ['identical', 'weak', 'strong']
    a_c = np.array([abs(SCENARIOS[s][0] - SCENARIOS[s][1]) for s in scenarios])
    panels = [('sel_W', r'(a)', 'Average number: $W$'),
              ('sel_D', r'(b)', 'Average number: $D$'),
              ('worst_loss', '(c)', 'worst-case quantile risk')]
    data = summary[summary.n == n]
    rows = data.set_index(['method', 'tau', 'metric', 'scenario']).sort_index()
    styles = [s for s in styles if s[0] in set(data.method)]
    # Offset the symbols slightly so coincident curves at 0 and 1 remain visible.
    #offsets = np.linspace(-.105, .105, len(styles))
    offsets = np.zeros(len(styles))

    with plt.rc_context({'font.family': 'serif',
                         'font.serif': ['Times New Roman', 'Liberation Serif', 'STIXGeneral', 'DejaVu Serif'],
                         'mathtext.fontset': 'cm', 'text.usetex': False,
                         'axes.grid': False, 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.05))
        for ax, (metric, title, ylabel) in zip(axes, panels):
            for offset, (method, tau, label, color, marker, line) in zip(offsets, styles):
                est = rows.loc[(method, tau, metric)].reindex(scenarios)
                ax.errorbar(a_c, est['mean'], yerr=est.mcse.fillna(0),
                            color=color, marker=marker, linestyle=line, linewidth=1.5,
                            markersize=5, capsize=2, elinewidth=.85, label=label)
            ax.set_xlim(-0.04, 0.64)
            ax.set_xticks(a_c, ['0', '0.1', '0.6'], fontsize=11)
            ax.set_xlabel(r'$a_c$', fontsize=14)
            ax.tick_params(axis='y', labelsize=11)
            ax.set_ylabel(ylabel, fontsize=14)
            if metric != 'worst_loss':
                ax.set_ylim(-.06, 1.06)
                ax.set_yticks(np.linspace(0, 1, 6))
            ax.text(.5, -.25, title, ha='center', va='top', transform=ax.transAxes, fontsize=15)

        # One framed legend for all three panels; do not cover the data.
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=True,
                   fancybox=False, framealpha=1, fontsize=11, columnspacing=1.4,
                   handlelength=2.8, bbox_to_anchor=(.5, .007))
        #fig.suptitle(r'Training environment contrast ($n=%d$ per environment)' % n, fontsize=16, y=.985)
        fig.subplots_adjust(left=.062, right=.99, top=.96, bottom=.327, wspace=.31)
        for ext in ('pdf', 'png'):
            fig.savefig(directory/f'heldout_n{n}.{ext}', dpi=200, bbox_inches='tight', pad_inches=.07)
        plt.close(fig)

def fresh_test(raw, seed, size):
    # A fresh pair of test sets, shared by all methods for this replication.
    eligible = raw.success.eq(1) & ~(raw.method.eq('EILLS') & raw.tau.ne(.5))
    for rep, sub in raw.loc[eligible].groupby('rep'):
        for e, a in enumerate(TEST_A):
            rng = np.random.default_rng([seed, 612841, int(rep), e])
            z,w = 2*rng.integers(0, 2, size=(2,size)) - 1
            draws = (z,w,rng.standard_normal(size),rng.random(size))
            cells = prepare_test(*environment(draws,a))
            values = score_test(cells,sub[['B0','BZ','BW','BD']].values,sub.tau.values)
            raw.loc[sub.index, [f'test{e}_{s}' for s in ('loss','cov','cov_wminus','cov_wplus')]] = values


def run_training(args, fit_dir, quick):
    cfg = dict(version=2, ns=args.n or ([200] if quick else [200,500,1000,2000]),
               reps=args.reps if args.reps is not None else (2 if quick else 200),
               test_n=args.test_n if args.test_n is not None else (1000 if quick else 10000),
               seed=args.seed, gamma=args.gamma, h=args.h, primary_only=args.primary_only,
               taus=[.5,.9], test_shifts=list(TEST_A), support_rule=RULE)
    if min(cfg['ns'])<2 or cfg['reps']<1 or cfg['test_n']<2 or cfg['gamma']<0 or (cfg['h'] is not None and cfg['h']<=0):
        raise ValueError('Training needs n>=2, positive reps and bandwidth, test-n>=2 and gamma>=0.')
    (fit_dir/'parts').mkdir(parents=True, exist_ok=True)
    manifest = fit_dir/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['settings'] != cfg:
        raise ValueError('Checkpoint settings differ. Choose a new --output, or use --mode report to rescore saved fits.')
    meta = dict(settings=cfg,train_shifts=SCENARIOS,coefficient_order=['intercept',*NAMES])
    manifest.write_text(json.dumps(meta,indent=2)+'\n',encoding='utf8')
    files = [fit_dir/'parts'/f'n{n:06d}_r{r:05d}.csv' for n in cfg['ns'] for r in range(cfg['reps'])]
    jobs = [(n,r,fit_dir,cfg) for n in cfg['ns'] for r in range(cfg['reps'])
            if not (fit_dir/'parts'/f'n{n:06d}_r{r:05d}.csv').exists()]
    workers = min(len(jobs) or 1, args.workers or min(4, os.cpu_count() or 1))
    def progress(iterator):
        for i,_ in enumerate(iterator,1):
            if i==1 or i%max(1,len(jobs)//10)==0 or i==len(jobs):
                print(f'Completed {len(files)-len(jobs)+i}/{len(files)}',flush=True)
    if workers==1:
        progress(map(one_rep,jobs))
    else:
        with ProcessPoolExecutor(workers) as pool: progress(pool.map(one_rep,jobs))
    parts = pd.concat([pd.read_csv(p,float_precision='round_trip') for p in files],ignore_index=True)
    raw = set_selection(parts).sort_values(['n','rep','scenario','tau','method'])
    raw.to_csv(fit_dir/'replicates.csv',index=False)
    save_fit_summary(raw,fit_dir)
    return raw, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode',choices=['auto','full','quick','report','fresh'],default='auto')
    parser.add_argument('--input',type=Path,help='existing results directory, replicates.csv or archive')
    parser.add_argument('--output','--out',type=Path,default=Path('iqr_environment_results'))
    parser.add_argument('--workers',type=int,default=0,help='training processes; 0 selects up to four')
    parser.add_argument('--n',type=int,nargs='+'); parser.add_argument('--reps',type=int)
    parser.add_argument('--seed',type=int,default=20261108); parser.add_argument('--test-seed',type=int,default=39107)
    parser.add_argument('--test-n',type=int); parser.add_argument('--gamma',type=float,default=20.); parser.add_argument('--h',type=float)
    parser.add_argument('--figure-n',type=int); parser.add_argument('--no-plot',action='store_true')
    parser.add_argument('--primary-only',action='store_true',help='omit optional baselines during fitting')
    args = parser.parse_args(); start = time.perf_counter()
    root = args.output; mode = args.mode
    source = args.input
    if source is None and mode in ('auto','report','fresh'):
        names = ('results','iqr_shift_results','IQR_shift_code_and_results.zip','IQR_heldout_evaluation.zip')
        choices = [base/name for name in names for base in (Path(__file__).resolve().parent,Path.cwd())]
        choices += [root/'full'/'fits',root/'quick'/'fits']
        source = next((p for p in choices if p.is_file() or (p/'replicates.csv').is_file()),None)
    if mode=='auto': mode = 'report' if source else 'quick'
    print(f'Mode: {mode}',flush=True)
    if mode in ('quick','full'):
        if args.input: parser.error('--input is for existing fits; use report or fresh.')
        source = root/mode/'fits'
        raw,meta = run_training(args,source,mode=='quick')
    else:
        if source is None: parser.error('Supply --input results (or first run --mode full).')
        raw,meta = read_saved(source)
        if mode=='report' and args.test_n is not None: parser.error('Use --mode fresh for new test data.')
        if mode=='fresh':
            size = args.test_n or meta['settings']['test_n']
            fresh_test(raw,args.test_seed,size)
    out = root/mode/'report'
    report(raw,out,args.figure_n,not args.no_plot)
    info = dict(mode=mode,input=str(source),support_rule=RULE,training_performed=mode in ('quick','full'),
                test_shifts=TEST_A,test_n_per_environment=args.test_n or meta['settings']['test_n'],
                evaluation_seed=args.test_seed if mode=='fresh' else meta['settings']['seed'],
                seconds=round(time.perf_counter()-start,2))
    (out/'evaluation_info.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf8')
    print(f'Report: {out} ({info["seconds"]} s).')
    if mode=='quick': print('Two-replicate default is for verification, not the paper.')


if __name__ == '__main__':
    main()
