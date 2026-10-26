import argparse, glob, os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('dataDir')
parser.add_argument('--split', dest='splitMode', default='perClass', choices=['strict', 'block', 'perClass', 'random'])
parser.add_argument('--trainFrac', dest='trainFrac', type=float, default=0.7)
parser.add_argument('--tierBalanced', dest='tierBalanced', action='store_true')
parser.add_argument('--calib', dest='calibMode', default='isotonic', choices=['isotonic', 'platt'])
parser.add_argument('--seeds', dest='numSeeds', type=int, default=5)
parser.add_argument('--boot', dest='numBoot', type=int, default=1000)
parser.add_argument('--tau', dest='tauDefault', type=float, default=1.0)
parser.add_argument('--out', dest='outDir', default='results')
args = parser.parse_args()
os.makedirs(args.outDir, exist_ok=True)

# severity tiers, my own mapping (assumption, stress test it later)
def getTier(label):
    l = label.lower()
    if l == 'benign': return 0
    if 'portscan' in l: return 1
    if any(k in l for k in ['infiltration', 'bot', 'heartbleed', 'sql']): return 3
    return 2  # dos, ddos, patators, web brute force, xss

def loadData(dataDir):
    frames = []
    for path in sorted(glob.glob(os.path.join(dataDir, '*.csv'))):
        df = pd.read_csv(path, encoding='latin1', low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df['session'] = os.path.basename(path)
        frames.append(df)
        print(f'loaded {os.path.basename(path)}: {len(df)} rows')
    return pd.concat(frames, ignore_index=True)

data = loadData(args.dataDir)
print('\nraw rows:', len(data))

data['Label'] = data['Label'].astype(str).str.strip().str.replace('ï¿½', '-', regex=False)
featCols = [c for c in data.columns if c not in ('Label', 'session')]
data[featCols] = data[featCols].apply(pd.to_numeric, errors='coerce')
data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=featCols)
print('after inf/nan drop:', len(data))
before = len(data)
data = data.drop_duplicates(subset=featCols + ['Label'])  # keeps file order
print(f'after dedup: {len(data)} (dropped {before - len(data)})')

data['tier'] = data['Label'].map(getTier)
data['isAttack'] = (data['tier'] >= 1).astype(int)

print('\nrows per session x label (after cleaning):')
print(data.groupby(['session', 'Label']).size().to_string())
print('\nrows per tier:')
print(data['tier'].value_counts().sort_index().to_string())

# split
day = data['session'].str.lower()
if args.splitMode == 'strict':
    isTest = day.str.startswith('thursday') | day.str.startswith('friday')
elif args.splitMode == 'random':
    # control: no time ordering, shows how optimistic a random split is
    isTest = np.random.RandomState(0).rand(len(data)) >= args.trainFrac
elif args.splitMode == 'perClass':
    # every attack type appears in both sets, test is the later part of each (session, label) stream
    pos = data.groupby(['session', 'Label']).cumcount()
    size = data.groupby(['session', 'Label'])['Label'].transform('size')
    isTest = (pos / size) >= args.trainFrac
else:
    # last 30% of each session (file order) is the held out block
    pos = data.groupby('session').cumcount()
    size = data.groupby('session')['Label'].transform('size')
    isTest = (pos / size) >= 0.7
trainPool = data[~isTest].reset_index(drop=True)
test = data[isTest].reset_index(drop=True)
print(f'\nsplit={args.splitMode}: trainPool={len(trainPool)}, test={len(test)}')
print('labels in test but not in train pool:',
      sorted(set(test['Label']) - set(trainPool['Label'])))
print('test label counts:')
print(test['Label'].value_counts().to_string())

xTest = test[featCols].astype(np.float32).values
tierTest = test['tier'].values

def calcEce(prob, y, nBins=10):
    bins = np.linspace(0, 1, nBins + 1)
    idx = np.clip(np.digitize(prob, bins) - 1, 0, nBins - 1)
    ece = 0.0
    for b in range(nBins):
        m = idx == b
        if m.sum() == 0: continue
        ece += m.mean() * abs(prob[m].mean() - y[m].mean())
    return ece

def calcMetrics(tier, esc):
    sev = tier >= 2  # should have been escalated
    tp = (esc & sev).sum()
    return {
        'precision': tp / max(esc.sum(), 1),
        'recall': tp / max(sev.sum(), 1),
        'falseEscRate': (esc & ~sev).sum() / max((~sev).sum(), 1),
        'tier3Recall': (esc & (tier == 3)).sum() / max((tier == 3).sum(), 1),
        'escPer10k': esc.mean() * 1e4,
    }

def fitCalibrator(score, y, mode):
    if mode == 'isotonic':
        iso = IsotonicRegression(out_of_bounds='clip').fit(score, y)
        return lambda s: iso.predict(s)
    lr = LogisticRegression().fit(np.log(np.clip(score, 1e-6, 1 - 1e-6) / (1 - np.clip(score, 1e-6, 1 - 1e-6))).reshape(-1, 1), y)
    return lambda s: lr.predict_proba(np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6))).reshape(-1, 1))[:, 1]

taus = np.round(np.arange(0.25, 3.01, 0.25), 2)
seedRows, sweepRows = [], []
store = {}

for seed in range(args.numSeeds):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(trainPool))
    nCal = int(0.2 * len(trainPool))
    cal = trainPool.iloc[perm[:nCal]]
    tr = trainPool.iloc[perm[nCal:]]
    xTr = tr[featCols].astype(np.float32).values
    xCal = cal[featCols].astype(np.float32).values

    # binary: is it an attack
    binModel = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, num_leaves=63,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                  random_state=seed, verbose=-1)
    binModel.fit(xTr, tr['isAttack'].values)
    rawCal = binModel.predict_proba(xCal)[:, 1]
    rawTest = binModel.predict_proba(xTest)[:, 1]
    calibrator = fitCalibrator(rawCal, cal['isAttack'].values, args.calibMode)
    pTest = calibrator(rawTest)

    # tier model, trained on attack rows only
    atk = tr[tr['tier'] >= 1]
    tierModel = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.1, num_leaves=31,
                                   class_weight='balanced' if args.tierBalanced else None,
                                   random_state=seed, verbose=-1)
    tierModel.fit(atk[featCols].astype(np.float32).values, atk['tier'].values)
    qTest = tierModel.predict_proba(xTest)
    expTier = qTest @ tierModel.classes_.astype(float)

    sHat = pTest * expTier
    sRaw = rawTest * expTier

    escBase = tierTest >= 1  # label derived baseline, optimistic
    escCal = sHat >= args.tauDefault
    escRaw = sRaw >= args.tauDefault

    for name, esc in [('signatureBaseline', escBase), ('gbtUncalibrated', escRaw), ('gbtCalibrated', escCal)]:
        row = calcMetrics(tierTest, esc)
        row.update({'system': name, 'seed': seed})
        seedRows.append(row)

    for t in taus:
        row = calcMetrics(tierTest, sHat >= t)
        row.update({'tau': t, 'seed': seed})
        sweepRows.append(row)

    seedRows[-1]['eceAttackProb'] = calcEce(pTest, (tierTest >= 1).astype(int))
    seedRows[-2]['eceAttackProb'] = calcEce(rawTest, (tierTest >= 1).astype(int))
    if seed == 0:
        store = {'pTest': pTest, 'rawTest': rawTest, 'sHat': sHat}
        print('\nmean expected tier on test, by true tier (seed 0):')
        print(pd.Series(expTier).groupby(tierTest).mean().round(3).to_string())
        print('train rows per label:')
        print(trainPool['Label'].value_counts().to_string())
        diag = pd.DataFrame({'label': test['Label'].values, 'p': pTest, 'expTier': expTier}).groupby('label').mean().round(3)
        print('\nmean attack prob and expected tier per label (seed 0):')
        print(diag.to_string())
    print(f'seed {seed} done')

seedDf = pd.DataFrame(seedRows)
summary = seedDf.groupby('system').agg(['mean', 'std']).drop(columns='seed')
pd.set_option('display.width', 200)
print(f'\n=== main table, tau={args.tauDefault}, calib={args.calibMode}, mean/std over {args.numSeeds} seeds ===')
print(summary.round(4).to_string())
summary.to_csv(os.path.join(args.outDir, 'mainTable.csv'))

sweepDf = pd.DataFrame(sweepRows).groupby('tau').mean().drop(columns='seed')
print('\n=== threshold sweep (mean over seeds) ===')
print(sweepDf.round(4).to_string())
sweepDf.to_csv(os.path.join(args.outDir, 'thresholdSweep.csv'))

# bootstrap CIs on seed 0 calibrated system
rng = np.random.RandomState(123)
bootRows = []
escSeed0 = store['sHat'] >= args.tauDefault
for _ in range(args.numBoot):
    idx = rng.randint(0, len(tierTest), len(tierTest))
    bootRows.append(calcMetrics(tierTest[idx], escSeed0[idx]))
bootDf = pd.DataFrame(bootRows)
print('\n=== bootstrap 95% CI, calibrated system, seed 0 ===')
print(pd.DataFrame({'lo': bootDf.quantile(0.025), 'hi': bootDf.quantile(0.975)}).round(4).to_string())

# per class recall (fraction escalated) at default tau, seed 0
perClass = test.assign(escCal=escSeed0, escBase=(tierTest >= 1)).groupby('Label').agg(
    n=('tier', 'size'), tier=('tier', 'first'), calibratedEscalated=('escCal', 'mean'),
    seenInTrain=('Label', lambda s: s.iloc[0] in set(trainPool['Label'])))
print('\n=== per class: fraction escalated (calibrated, seed 0) ===')
print(perClass.round(4).to_string())
perClass.to_csv(os.path.join(args.outDir, 'perClass.csv'))

# reliability plot for attack probability
y = (tierTest >= 1).astype(int)
fig, ax = plt.subplots(figsize=(4, 4))
for name, p in [('uncalibrated', store['rawTest']), ('calibrated', store['pTest'])]:
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    xs = [p[idx == b].mean() for b in range(10) if (idx == b).sum() > 0]
    ys = [y[idx == b].mean() for b in range(10) if (idx == b).sum() > 0]
    ax.plot(xs, ys, marker='o', label=name)
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel('predicted attack prob'); ax.set_ylabel('observed attack rate'); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(args.outDir, 'reliability.png'), dpi=200)

# sweep plot
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(sweepDf.index, sweepDf['tier3Recall'], marker='o', label='tier 3 recall')
ax.plot(sweepDf.index, sweepDf['recall'], marker='o', label='severe recall')
ax2 = ax.twinx()
ax2.plot(sweepDf.index, sweepDf['escPer10k'], color='gray', ls='--', label='escalations / 10k')
ax.set_xlabel('threshold tau'); ax.set_ylabel('recall'); ax2.set_ylabel('escalations / 10k')
ax.legend(loc='lower left'); fig.tight_layout()
fig.savefig(os.path.join(args.outDir, 'thresholdSweep.png'), dpi=200)
print(f'\nsaved csvs and plots to {args.outDir}/')
