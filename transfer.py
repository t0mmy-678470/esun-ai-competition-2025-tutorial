"""
2025玉山人工智慧挑戰賽範例程式碼 - 改進版
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


def LoadCSV(dir_path):
    """
    讀取挑戰賽提供的3個資料集：交易資料、警示帳戶註記、待預測帳戶清單
    """
    df_txn = pd.read_csv(os.path.join(dir_path, 'acct_transaction.csv'))
    df_alert = pd.read_csv(os.path.join(dir_path, 'acct_alert.csv'))
    df_test = pd.read_csv(os.path.join(dir_path, 'acct_predict.csv'))
    
    print("(Finish) Load Dataset.")
    print(f"Transaction records: {len(df_txn)}")
    print(f"Alert accounts: {len(df_alert)}")
    print(f"Test accounts: {len(df_test)}")
    return df_txn, df_alert, df_test


def PreProcessing(df):
    """
    增強的特徵工程
    """
    # 先複製一份避免修改原始資料
    df = df.copy()
    
    # 轉換時間欄位為數值型
    df['txn_time'] = pd.to_numeric(df['txn_time'], errors='coerce').fillna(0).astype(int)
    df['txn_date'] = pd.to_numeric(df['txn_date'], errors='coerce').fillna(0).astype(int)
    
    features_list = []
    
    # 1. 基本統計量 - 匯款方
    send_stats = df.groupby('from_acct').agg({
        'txn_amt': ['sum', 'mean', 'std', 'max', 'min', 'count'],
        'to_acct': 'nunique'  # 匯款給多少不同帳戶
    }).reset_index()
    send_stats.columns = ['acct', 'total_send_amt', 'avg_send_amt', 'std_send_amt', 
                          'max_send_amt', 'min_send_amt', 'send_count', 'unique_recv_accts']
    features_list.append(send_stats)
    
    # 2. 基本統計量 - 收款方
    recv_stats = df.groupby('to_acct').agg({
        'txn_amt': ['sum', 'mean', 'std', 'max', 'min', 'count'],
        'from_acct': 'nunique'  # 從多少不同帳戶收款
    }).reset_index()
    recv_stats.columns = ['acct', 'total_recv_amt', 'avg_recv_amt', 'std_recv_amt',
                          'max_recv_amt', 'min_recv_amt', 'recv_count', 'unique_send_accts']
    features_list.append(recv_stats)
    
    # 3. 交易時間特徵
    df['hour'] = df['txn_time'] // 10000
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    df['is_weekend'] = (df['txn_date'] % 7 >= 5).astype(int)
    
    # 夜間交易統計
    night_send = df[df['is_night']==1].groupby('from_acct').size().rename('night_send_count')
    night_recv = df[df['is_night']==1].groupby('to_acct').size().rename('night_recv_count')
    
    # 週末交易統計
    weekend_send = df[df['is_weekend']==1].groupby('from_acct').size().rename('weekend_send_count')
    weekend_recv = df[df['is_weekend']==1].groupby('to_acct').size().rename('weekend_recv_count')
    
    # 4. 自我交易特徵
    self_txn_send = df[df['is_self_txn']=='Y'].groupby('from_acct').size().rename('self_txn_count')
    self_txn_recv = df[df['is_self_txn']=='Y'].groupby('to_acct').size().rename('self_txn_recv_count')
    
    # 5. 跨行交易特徵
    cross_bank_send = df[df['to_acct_type']=='02'].groupby('from_acct').size().rename('cross_bank_send_count')
    cross_bank_recv = df[df['from_acct_type']=='02'].groupby('to_acct').size().rename('cross_bank_recv_count')
    
    # 6. 幣別特徵 - 外幣交易
    foreign_currency_send = df[df['currency_type']!='TWD'].groupby('from_acct').size().rename('foreign_send_count')
    foreign_currency_recv = df[df['currency_type']!='TWD'].groupby('to_acct').size().rename('foreign_recv_count')
    
    # 7. 交易通路多樣性
    channel_diversity_send = df.groupby('from_acct')['channel_type'].nunique().rename('channel_diversity_send')
    channel_diversity_recv = df.groupby('to_acct')['channel_type'].nunique().rename('channel_diversity_recv')
    
    # 8. 時間相關特徵
    first_txn_send = df.groupby('from_acct')['txn_date'].min().rename('first_send_date')
    last_txn_send = df.groupby('from_acct')['txn_date'].max().rename('last_send_date')
    first_txn_recv = df.groupby('to_acct')['txn_date'].min().rename('first_recv_date')
    last_txn_recv = df.groupby('to_acct')['txn_date'].max().rename('last_recv_date')
    
    # 9. 集中度特徵 - 是否主要與少數帳戶交易
    top3_send_ratio = df.groupby(['from_acct', 'to_acct'])['txn_amt'].sum().groupby('from_acct').apply(
        lambda x: x.nlargest(3).sum() / x.sum() if x.sum() > 0 else 0
    ).rename('top3_send_concentration')
    
    top3_recv_ratio = df.groupby(['to_acct', 'from_acct'])['txn_amt'].sum().groupby('to_acct').apply(
        lambda x: x.nlargest(3).sum() / x.sum() if x.sum() > 0 else 0
    ).rename('top3_recv_concentration')
    
    # 10. 金額相關進階特徵
    # 大額交易 (超過平均的2倍)
    mean_amt = df['txn_amt'].mean()
    large_txn_send = df[df['txn_amt'] > mean_amt * 2].groupby('from_acct').size().rename('large_send_count')
    large_txn_recv = df[df['txn_amt'] > mean_amt * 2].groupby('to_acct').size().rename('large_recv_count')
    
    # 小額交易
    small_txn_send = df[df['txn_amt'] < mean_amt * 0.1].groupby('from_acct').size().rename('small_send_count')
    small_txn_recv = df[df['txn_amt'] < mean_amt * 0.1].groupby('to_acct').size().rename('small_recv_count')
    
    # 整合所有特徵
    df_result = pd.concat(features_list, axis=1)
    
    # 合併其他特徵
    for feature in [night_send, night_recv, weekend_send, weekend_recv,
                    self_txn_send, self_txn_recv, cross_bank_send, cross_bank_recv,
                    foreign_currency_send, foreign_currency_recv, 
                    channel_diversity_send, channel_diversity_recv,
                    first_txn_send, last_txn_send, first_txn_recv, last_txn_recv,
                    top3_send_ratio, top3_recv_ratio,
                    large_txn_send, large_txn_recv, small_txn_send, small_txn_recv]:
        df_result = df_result.join(feature, how='left')
    
    # 處理重複的 acct 欄位
    df_result = df_result.loc[:,~df_result.columns.duplicated()]
    
    # 填充缺失值
    df_result = df_result.fillna(0)
    
    # 添加帳戶類型
    df_from = df[['from_acct', 'from_acct_type']].rename(columns={'from_acct': 'acct', 'from_acct_type': 'is_esun'})
    df_to = df[['to_acct', 'to_acct_type']].rename(columns={'to_acct': 'acct', 'to_acct_type': 'is_esun'})
    df_acc = pd.concat([df_from, df_to], ignore_index=True).drop_duplicates().reset_index(drop=True)
    
    df_result = pd.merge(df_result, df_acc, on='acct', how='left')
    
    # 衍生特徵
    df_result['send_recv_ratio'] = df_result['total_send_amt'] / (df_result['total_recv_amt'] + 1)
    df_result['send_recv_count_ratio'] = df_result['send_count'] / (df_result['recv_count'] + 1)
    df_result['avg_txn_per_acct'] = df_result['total_send_amt'] / (df_result['unique_recv_accts'] + 1)
    df_result['txn_span'] = df_result['last_send_date'] - df_result['first_send_date'] + 1
    df_result['txn_frequency'] = df_result['send_count'] / (df_result['txn_span'] + 1)
    df_result['night_ratio'] = df_result['night_send_count'] / (df_result['send_count'] + 1)
    df_result['cross_bank_ratio'] = df_result['cross_bank_send_count'] / (df_result['send_count'] + 1)
    df_result['large_txn_ratio'] = df_result['large_send_count'] / (df_result['send_count'] + 1)
    
    print("(Finish) PreProcessing with enhanced features.")
    print(f"Total features: {len(df_result.columns)-2}")  # -2 for 'acct' and 'is_esun'
    return df_result

def TrainTestSplit(df, df_alert, df_test):
    """
    切分訓練集及測試集
    """
    X_train = df[(~df['acct'].isin(df_test['acct'])) & (df['is_esun']=='01')].drop(columns=['is_esun']).copy()
    y_train = X_train['acct'].isin(df_alert['acct']).astype(int)
    X_test = df[df['acct'].isin(df_test['acct'])].drop(columns=['is_esun']).copy()
    
    print(f"(Finish) Train-Test-Split")
    print(f"Training samples: {len(X_train)}")
    print(f"Alert accounts in training: {y_train.sum()}")
    print(f"Non-alert accounts in training: {len(y_train) - y_train.sum()}")
    print(f"Alert ratio: {y_train.sum()/len(y_train):.4f}")
    print(f"Test samples: {len(X_test)}")
    
    return X_train, X_test, y_train

def Modeling(X_train, y_train, X_test):
    """
    改進的模型訓練，使用 SMOTE 處理類別不平衡
    """
    # 分離帳戶ID
    train_accts = X_train['acct'].values
    test_accts = X_test['acct'].values
    
    X_train_features = X_train.drop(columns=['acct'])
    X_test_features = X_test.drop(columns=['acct'])
    
    # 處理無限值和 NaN
    X_train_features = X_train_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test_features = X_test_features.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)
    
    # 處理類別不平衡 - SMOTE
    print("Applying SMOTE for class imbalance...")
    try:
        smote = SMOTE(random_state=42, k_neighbors=min(5, y_train.sum()-1))
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        print(f"After SMOTE - Training samples: {len(X_train_balanced)}")
        print(f"After SMOTE - Alert: {y_train_balanced.sum()}, Non-alert: {len(y_train_balanced) - y_train_balanced.sum()}")
    except Exception as e:
        print(f"SMOTE failed: {e}, using original data with class_weight='balanced'")
        X_train_balanced = X_train_scaled
        y_train_balanced = y_train
    
    # 使用 Random Forest，對不平衡資料效果較好且較穩定
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train_balanced, y_train_balanced)
    
    # 預測
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # 調整閾值以優化 F1-score
    # 由於警示帳戶很少，需要較低的閾值
    threshold = 0.2  # 可以調整此閾值
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    print(f"(Finish) Modeling")
    print(f"Predicted alerts: {y_pred.sum()}")
    print(f"Predicted alert ratio: {y_pred.sum()/len(y_pred):.4f}")
    
    # 特徵重要性
    feature_importance = pd.DataFrame({
        'feature': X_train_features.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 15 Important Features:")
    print(feature_importance.head(15))
    
    return y_pred

def OutputCSV(path, df_test, X_test, y_pred):
    """
    輸出預測結果
    """
    df_pred = pd.DataFrame({
        'acct': X_test['acct'].values,
        'label': y_pred
    })
    
    df_out = df_test[['acct']].merge(df_pred, on='acct', how='left')
    df_out.to_csv(path, index=False)
    
    print(f"(Finish) Output saved to {path}")
    print(f"Total predictions: {len(df_out)}")
    print(f"Predicted alert accounts: {y_pred.sum()}")
    print(f"Predicted alert rate: {y_pred.sum() / len(y_pred):.2%}")

if __name__ == "__main__":
    dir_path = "../preliminary_data/"
    df_txn, df_alert, df_test = LoadCSV(dir_path)
    df_X = PreProcessing(df_txn)
    X_train, X_test, y_train = TrainTestSplit(df_X, df_alert, df_test)
    y_pred = Modeling(X_train, y_train, X_test)
    out_path = "result.csv"
    OutputCSV(out_path, df_test, X_test, y_pred)