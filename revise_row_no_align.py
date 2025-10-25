"""
2025 玉山人工智慧挑戰賽 

"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier  # ★



def LoadCSV(dir_path):
    df_txn   = pd.read_csv(os.path.join(dir_path, 'acct_transaction.csv'))
    df_alert = pd.read_csv(os.path.join(dir_path, 'acct_alert.csv'))
    df_test  = pd.read_csv(os.path.join(dir_path, 'acct_predict.csv'))
    print("(Finish) Load Dataset.")
    print(f"Transaction records: {len(df_txn)}")
    print(f"Alert accounts: {len(df_alert)}")
    print(f"Test accounts: {len(df_test)}")
    return df_txn, df_alert, df_test

# 特徵工程（全部以 acct 為 key）
def PreProcessing(df_txn: pd.DataFrame, df_alert: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    df = df_txn.copy()

    # 型別清理
    df['txn_amt']  = pd.to_numeric(df['txn_amt'],  errors='coerce').fillna(0.0)
    df['txn_time'] = pd.to_numeric(df['txn_time'], errors='coerce').fillna(0).astype(int)
    df['txn_date'] = pd.to_numeric(df['txn_date'], errors='coerce').fillna(0).astype(int)

    # ★ base_accts：確保每個帳戶只出現一次，且含 test/alert 的帳戶
    base_accts = pd.DataFrame({'acct': pd.unique(
        pd.concat([
            df['from_acct'], df['to_acct'],
            df_alert['acct'], df_test['acct']
        ], ignore_index=True).dropna()
    )})

    # ===== 送方統計 =====
    send = df.groupby('from_acct').agg(
        total_send_amt=('txn_amt','sum'),
        avg_send_amt=('txn_amt','mean'),
        std_send_amt=('txn_amt','std'),
        max_send_amt=('txn_amt','max'),
        min_send_amt=('txn_amt','min'),
        send_count=('txn_amt','count'),
        unique_recv_accts=('to_acct','nunique')
    ).reset_index().rename(columns={'from_acct':'acct'})

    # ===== 收方統計 =====
    recv = df.groupby('to_acct').agg(
        total_recv_amt=('txn_amt','sum'),
        avg_recv_amt=('txn_amt','mean'),
        std_recv_amt=('txn_amt','std'),
        max_recv_amt=('txn_amt','max'),
        min_recv_amt=('txn_amt','min'),
        recv_count=('txn_amt','count'),
        unique_send_accts=('from_acct','nunique')
    ).reset_index().rename(columns={'to_acct':'acct'})

    # ===== 時間特徵（先在明細上標記，再 groupby）=====
    df['hour'] = df['txn_time']//10000
    df['is_night']   = ((df['hour']>=22) | (df['hour']<=6)).astype(int)
    df['is_weekend'] = (df['txn_date']%7 >= 5).astype(int)

    night_send   = df.groupby('from_acct')['is_night'].sum().reset_index().rename(columns={'from_acct':'acct','is_night':'night_send_count'})
    night_recv   = df.groupby('to_acct')['is_night'].sum().reset_index().rename(columns={'to_acct':'acct','is_night':'night_recv_count'})
    weekend_send = df.groupby('from_acct')['is_weekend'].sum().reset_index().rename(columns={'from_acct':'acct','is_weekend':'weekend_send_count'})
    weekend_recv = df.groupby('to_acct')['is_weekend'].sum().reset_index().rename(columns={'to_acct':'acct','is_weekend':'weekend_recv_count'})

    # ===== 自我 / 跨行 / 幣別 =====
    self_send = df[df['is_self_txn']=='Y'].groupby('from_acct').size().reset_index(name='self_txn_count').rename(columns={'from_acct':'acct'})
    self_recv = df[df['is_self_txn']=='Y'].groupby('to_acct').size().reset_index(name='self_txn_recv_count').rename(columns={'to_acct':'acct'})

    cross_send = df[df['to_acct_type']=='02'].groupby('from_acct').size().reset_index(name='cross_bank_send_count').rename(columns={'from_acct':'acct'})
    cross_recv = df[df['from_acct_type']=='02'].groupby('to_acct').size().reset_index(name='cross_bank_recv_count').rename(columns={'to_acct':'acct'})

    fx_send = df[df['currency_type']!='TWD'].groupby('from_acct').size().reset_index(name='foreign_send_count').rename(columns={'from_acct':'acct'})
    fx_recv = df[df['currency_type']!='TWD'].groupby('to_acct').size().reset_index(name='foreign_recv_count').rename(columns={'to_acct':'acct'})

    # ===== 通路多樣性 =====
    ch_send = df.groupby('from_acct')['channel_type'].nunique().reset_index().rename(columns={'from_acct':'acct','channel_type':'channel_diversity_send'})
    ch_recv = df.groupby('to_acct')['channel_type'].nunique().reset_index().rename(columns={'to_acct':'acct','channel_type':'channel_diversity_recv'})

    # ===== 最早/最晚日期 =====
    first_send = df.groupby('from_acct')['txn_date'].min().reset_index().rename(columns={'from_acct':'acct','txn_date':'first_send_date'})
    last_send  = df.groupby('from_acct')['txn_date'].max().reset_index().rename(columns={'from_acct':'acct','txn_date':'last_send_date'})
    first_recv = df.groupby('to_acct')['txn_date'].min().reset_index().rename(columns={'to_acct':'acct','txn_date':'first_recv_date'})
    last_recv  = df.groupby('to_acct')['txn_date'].max().reset_index().rename(columns={'to_acct':'acct','txn_date':'last_recv_date'})

    # ===== 集中度（top3 收/付比）=====
    send_pair = df.groupby(['from_acct','to_acct'])['txn_amt'].sum().reset_index()
    top3_send = (send_pair
                 .sort_values(['from_acct','txn_amt'], ascending=[True, False])
                 .groupby('from_acct')
                 .apply(lambda x: x['txn_amt'].nlargest(3).sum()/x['txn_amt'].sum() if x['txn_amt'].sum()>0 else 0.0)
                 .reset_index(name='top3_send_concentration')
                 .rename(columns={'from_acct':'acct'}))

    recv_pair = df.groupby(['to_acct','from_acct'])['txn_amt'].sum().reset_index()
    top3_recv = (recv_pair
                 .sort_values(['to_acct','txn_amt'], ascending=[True, False])
                 .groupby('to_acct')
                 .apply(lambda x: x['txn_amt'].nlargest(3).sum()/x['txn_amt'].sum() if x['txn_amt'].sum()>0 else 0.0)
                 .reset_index(name='top3_recv_concentration')
                 .rename(columns={'to_acct':'acct'}))

    # ===== 大/小額 =====
    mean_amt = df['txn_amt'].mean() if len(df) else 0.0
    large_send = df[df['txn_amt'] > mean_amt*2].groupby('from_acct').size().reset_index(name='large_send_count').rename(columns={'from_acct':'acct'})
    large_recv = df[df['txn_amt'] > mean_amt*2].groupby('to_acct').size().reset_index(name='large_recv_count').rename(columns={'to_acct':'acct'})
    small_send = df[df['txn_amt'] < max(mean_amt*0.1, 1e-9)].groupby('from_acct').size().reset_index(name='small_send_count').rename(columns={'from_acct':'acct'})
    small_recv = df[df['txn_amt'] < max(mean_amt*0.1, 1e-9)].groupby('to_acct').size().reset_index(name='small_recv_count').rename(columns={'to_acct':'acct'})

    # ===== 帳戶類型（01/02）：兩側來源取「最小值=01 優先」=====
    acc_from = df[['from_acct','from_acct_type']].dropna().rename(columns={'from_acct':'acct','from_acct_type':'acct_type'})
    acc_to   = df[['to_acct','to_acct_type']].dropna().rename(columns={'to_acct':'acct','to_acct_type':'acct_type'})
    acc_typ  = pd.concat([acc_from, acc_to], ignore_index=True)
    acc_typ['acct_type'] = pd.to_numeric(acc_typ['acct_type'], errors='coerce').fillna(99).astype(int)
    acc_typ  = acc_typ.groupby('acct', as_index=False)['acct_type'].min()  # 01 優先
    acc_typ['acct_type'] = acc_typ['acct_type'].astype(str).str.zfill(2)

    # ★ 把所有特徵「依 acct」左連到 base_accts
    feat = base_accts.copy()
    for tbl in [send, recv, night_send, night_recv, weekend_send, weekend_recv,
                self_send, self_recv, cross_send, cross_recv, fx_send, fx_recv,
                ch_send, ch_recv, first_send, last_send, first_recv, last_recv,
                top3_send, top3_recv, large_send, large_recv, small_send, small_recv]:
        feat = feat.merge(tbl, on='acct', how='left')

    # 衍生比率
    for col in ['total_send_amt','total_recv_amt','send_count','recv_count',
                'night_send_count','cross_bank_send_count','large_send_count',
                'unique_recv_accts','last_send_date','first_send_date']:
        if col not in feat.columns: feat[col] = 0.0

    feat['send_recv_ratio']         = feat['total_send_amt']/(feat['total_recv_amt']+1)
    feat['send_recv_count_ratio']   = feat['send_count']/(feat['recv_count']+1)
    feat['avg_txn_per_acct']        = feat['total_send_amt']/(feat['unique_recv_accts']+1)
    feat['txn_span']                = (feat['last_send_date']-feat['first_send_date']).clip(lower=0)
    feat['txn_frequency']           = feat['send_count']/(feat['txn_span']+1)
    feat['night_ratio']             = feat['night_send_count']/(feat['send_count']+1)
    feat['cross_bank_ratio']        = feat['cross_bank_send_count']/(feat['send_count']+1)
    feat['large_txn_ratio']         = feat['large_send_count']/(feat['send_count']+1)

    # 補帳戶類型（可能有帳戶沒出現在交易中）
    feat = feat.merge(acc_typ, on='acct', how='left')
    feat['acct_type'] = feat['acct_type'].fillna('01')  # 預設玉山（保守）

    # 清理 NA/inf
    feat = feat.replace([np.inf,-np.inf], np.nan).fillna(0.0)

    print("(Finish) PreProcessing")
    print(f"Feature table rows (unique accts): {len(feat)}")
    return feat


# 切訓練 / 測試
def TrainTestSplit(feat: pd.DataFrame, df_alert: pd.DataFrame, df_test: pd.DataFrame):
    # 以玉山帳戶為訓練（01）
    train_df = feat[(feat['acct_type']=='01') & (~feat['acct'].isin(df_test['acct']))].copy()
    y = train_df['acct'].isin(df_alert['acct']).astype(int).values

    test_df  = feat[feat['acct'].isin(df_test['acct'])].copy()

    print(f"(Split) Train rows: {len(train_df)} | Positives: {y.sum()} | Ratio={y.mean():.4f}")
    print(f"(Split) Test rows:  {len(test_df)} (should match df_test: {len(df_test)})")
    return train_df, y, test_df


# 訓練 + 閾值掃描
def Modeling(train_df, y, test_df):
    # 拿掉識別欄
    drop_cols = ['acct','acct_type']
    X = train_df.drop(columns=drop_cols, errors='ignore').values
    X_test = test_df.drop(columns=drop_cols, errors='ignore').values

    # 驗證集
    if y.sum()==0:
        print("No positive in training, fallback to all-zero prediction.")
        return np.zeros(len(test_df), dtype=int)

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 模型：BalancedRandomForest（若失敗則退回 RF）
    try:
        clf = BalancedRandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            max_features='sqrt',
            sampling_strategy='auto',
            random_state=42,
            n_jobs=-1
        )
    except Exception:
        clf = RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            max_features='sqrt',
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )

    clf.fit(X_tr, y_tr)

    # 驗證找最佳 threshold
    va_proba = clf.predict_proba(X_va)[:,1]
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.05, 0.95, 19):
        f1 = f1_score(y_va, (va_proba>=t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"(Validate) best F1={best_f1:.4f} @ threshold={best_t:.2f}")

    # 用全訓練重訓
    clf.fit(X, y)
    test_proba = clf.predict_proba(X_test)[:,1]
    y_pred = (test_proba >= best_t).astype(int)
    print(f"(Predict) positive ratio on test: {y_pred.mean():.4f} (count={y_pred.sum()})")
    return y_pred


def OutputCSV(path, df_test, test_df, y_pred):
    out = pd.DataFrame({'acct': test_df['acct'].values, 'label': y_pred.astype(int)})
    # 保證和 df_test 的 acct 一一對齊
    out = df_test[['acct']].merge(out, on='acct', how='left')
    out['label'] = out['label'].fillna(0).astype(int)
    out.to_csv(path, index=False)
    print(f"(Finish) Output saved to {path} | rows={len(out)} | positives={out['label'].sum()}")


if __name__ == "__main__":
    dir_path = "../preliminary_data/"  
    df_txn, df_alert, df_test = LoadCSV(dir_path)

    feat = PreProcessing(df_txn, df_alert, df_test)           # ★ 這裡已經保證 key 對齊、含 test/alert
    train_df, y, test_df = TrainTestSplit(feat, df_alert, df_test)

    y_pred = Modeling(train_df, y, test_df)                    # ★ 內含驗證調閾值
    OutputCSV("result.csv", df_test, test_df, y_pred)
