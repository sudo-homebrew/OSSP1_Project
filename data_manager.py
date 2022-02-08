import pandas as pd
import numpy as np

COLUMNS_CHART_DATA = ['date', 'time', 'open', 'high', 'low', 'close', 'volume', 'pc']

COLUMNS_CHART_DATA_L = ['time', 'open', 'high', 'low', 'close', 'volume', 'pc']

COLUMNS_CHART_DATA_T = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']

COLUMNS_CHART_DATA_N = ['time', 'open', 'high', 'low', 'close', 'volume']

COLUMNS_CHART_DATA_K = ['close']

COLUMNS_CHART_DATA_OSSP_ALL = ['next_close_ratio']

COLUMNS_TRAINING_DATA_OSSP_ALL = [
    'open_lastopen_ratio', 'high_lasthigh_ratio', 'low_lastlow_ratio',
    'close_lastclose_ratio', 'volume_lastvolume_ratio',
    'open_ma10_ratio', 'high_ma10_ratio', 'low_ma10_ratio', 'close_ma10_ratio', 'volume_ma10_ratio',
    'percent_b', 'Band_width',
    'PDI', 'MDI', 'ADX',
    'percent_k', 'percent_d',
    'MACD', 'SIGNAL',
    'RSI', 'Momentum'
]

def make_dataset(data, label, window_size=5):
    feature_list = []
    label_list = []
    for i in range(len(data) - window_size):
        feature_list.append(np.array(data.iloc[i:i+window_size]))
        label_list.append(np.array(label.iloc[i+window_size]))
    return np.array(feature_list), np.array(label_list)

def preprocess(input_data, ver):
    data = input_data
    # 시작 ~ 5분까지의 시고저종 거래량 평균으로 기준점을 잡고
    # 각 입력 데이터(시고저종 거래량을) v라 할때 각 훈련 데이터 = v - 기준점 / 기준점으로 비율로 나타내면 될듯?
    # 일봉 데이터 관찰 -> 5분 이동평균선이 주가를 잘 따르는 것 같음(경향성의 반영)
    if ver == 'ossp_all':
        # 주가, 거래량 이동 평균
        windows0 = [10]  # 이동평균선 5분짜리만 사용 필요하다면 windows = [5, 10, 20, 60, 120] 추가 가능
        for window in windows0:
            data['open_ma{}'.format(window)] = data['open'].rolling(window).mean()
            data['high_ma{}'.format(window)] = data['high'].rolling(window).mean()
            data['low_ma{}'.format(window)] = data['low'].rolling(window).mean()
            data['close_ma{}'.format(window)] = data['close'].rolling(window).mean()
            data['volume_ma{}'.format(window)] = data['volume'].rolling(window).mean()
            # 주가, 거래량 이동 평균 비율
            data['open_ma%d_ratio' % window] = (data['open'] - data['open_ma%d' % window]) / data['open_ma%d' % window]
            data['high_ma%d_ratio' % window] = (data['high'] - data['high_ma%d' % window]) / data['high_ma%d' % window]
            data['low_ma%d_ratio' % window] = (data['low'] - data['low_ma%d' % window]) / data['low_ma%d' % window]
            data['close_ma%d_ratio' % window] = (data['close'] - data['close_ma%d' % window]) / data['close_ma%d' % window]
            data['volume_ma%d_ratio' % window] = (data['volume'] - data['volume_ma%d' % window]) / data['volume_ma%d' % window]
        windows1 = [20]
        # BollingerBand
        for window in windows1:
            # 중심선 # 20개 표준편차 # 하한선 # 상한선
            data['close_ma{}'.format(window)] = data['close'].rolling(window).mean()
            data['close_std{}'.format(window)] = data['close'].rolling(window).std()
            data['down_line'] = data['close_ma{}'.format(window)] - data['close_std{}'.format(window)] * 2
            data['upper_line'] = data['close_ma{}'.format(window)] + data['close_std{}'.format(window)] * 2
            data['percent_b'] = (data['close'] - data['down_line']) / (data['upper_line'] - data['down_line'])
            data['Band_width'] = (data['upper_line'] - data['down_line']) / data['close_ma{}'.format(window)]
        # DMI
        data['PDM'] = np.zeros(len(data))
        data.loc[1:, 'PDM'] = np.abs(data.loc[1:, 'high'].copy().values - data.loc[:len(data) - 2, 'high'].copy().values)
        data['MDM'] = np.zeros(len(data))
        data.loc[1:, 'MDM'] = np.abs(data.loc[1:, 'low'].copy().values - data.loc[:len(data) - 2, 'low'].copy().values)
        data['TR'] = np.zeros(len(data))
        data.loc[1:, 'TR'] = np.maximum(np.abs(data.loc[1:, 'high'].copy().values - data.loc[1:, 'low'].copy().values),
                                    np.abs(data.loc[1:, 'high'].copy().values - data.loc[:len(data) - 2, 'close'].copy().values),
                                    np.abs(data.loc[1:, 'low'].copy().values - data.loc[:len(data) - 2, 'close'].copy().values))
        windows2 = [10]
        for window in windows2:
            data['PDI'] = data['PDM'].rolling(window).mean() / data['TR'].rolling(window).mean()
            data['MDI'] = data['MDM'].rolling(window).mean() / data['TR'].rolling(window).mean()
        data['ADX'] = np.abs(data['PDI'] - data['MDI']) / (data['PDI'] + data['MDI']) * 100
        # STOCHASTIC
        windows3 = [5]
        for window in windows3:
            data['percent_k'] = (data['close'] - data['close'].rolling(window).min()) / (data['close'].rolling(window).max() - data['close'].rolling(window).min())
        windows4 = [3]
        for window in windows4:
            data['percent_d'] = data['percent_k'].rolling(window).mean()
        # MACD
        data['ema12'] = data['close'].ewm(12).mean()
        data['ema26'] = data['close'].ewm(26).mean()
        data['MACD'] = data['ema12'] - data['ema26']
        data['SIGNAL'] = data['MACD'].ewm(9).mean()
        # RSI
        windows5 = [14]
        data['U'] = np.where(data['close'].diff(1) > 0, data['close'].diff(1), 0)
        data['D'] = np.where(data['close'].diff(1) < 0, data['close'].diff(1) * (-1), 0)
        for window in windows5:
            data['RSI'] = data['U'].rolling(window).mean() / (data['U'].rolling(window).mean() + data['D'].rolling(window).mean())
        # Momentum
        data['Momentum'] = np.zeros(len(data))
        data.loc[10:, 'Momentum'] = (data.loc[10:, 'close'].copy().values / data.loc[:len(data)-11, 'close'].copy().values) * 100

        # 주가, 거래량 전 대비 비율
        data['open_lastopen_ratio'] = np.zeros(len(data))
        data.loc[1:, 'open_lastopen_ratio'] = (data.loc[1:, 'open'].copy().values - data.loc[:len(data)-2, 'open'].copy().values) / data.loc[:len(data)-2, 'open'].copy().values
        data['high_lasthigh_ratio'] = np.zeros(len(data))
        data.loc[1:, 'high_lasthigh_ratio'] = (data.loc[1:, 'high'].copy().values - data.loc[:len(data)-2, 'high'].copy().values) / data.loc[:len(data)-2, 'high'].copy().values
        data['low_lastlow_ratio'] = np.zeros(len(data))
        data.loc[1:, 'low_lastlow_ratio'] = (data.loc[1:, 'low'].copy().values - data.loc[:len(data)-2, 'low'].copy().values) / data.loc[:len(data)-2, 'low'].copy().values
        data['close_lastclose_ratio'] = np.zeros(len(data))
        data.loc[1:, 'close_lastclose_ratio'] = (data.loc[1:, 'close'].copy().values - data.loc[:len(data)-2, 'close'].copy().values) / data.loc[:len(data)-2, 'close'].copy().values
        data['volume_lastvolume_ratio'] = np.zeros(len(data))
        data.loc[1:, 'volume_lastvolume_ratio'] = (data.loc[1:, 'volume'].copy().values - data.loc[:len(data)-2, 'volume'].copy().values) / data.loc[:len(data)-2, 'volume'].copy().replace(
            to_replace=0, method='ffill').replace(to_replace=0, method='bfill').values
        data['next_close_ratio'] = np.zeros(len(data))
        data.loc[:len(data)-2, 'next_close_ratio'] = (data.loc[1:, 'close'].copy().values - data.loc[:len(data)-2, 'close'].copy().values) / data.loc[:len(data)-2, 'close'].copy().values * 100
    if ver == 'ossp_lstm':
        # 주가, 거래량 이동 평균
        windows0 = [10]  # 이동평균선 5분짜리만 사용 필요하다면 windows = [5, 10, 20, 60, 120] 추가 가능
        for window in windows0:
            data['open_ma{}'.format(window)] = data['open'].rolling(window).mean()
            data['high_ma{}'.format(window)] = data['high'].rolling(window).mean()
            data['low_ma{}'.format(window)] = data['low'].rolling(window).mean()
            data['close_ma{}'.format(window)] = data['close'].rolling(window).mean()
            data['volume_ma{}'.format(window)] = data['volume'].rolling(window).mean()
            # 주가, 거래량 이동 평균 비율
            data['open_ma%d_ratio' % window] = (data['open'] - data['open_ma%d' % window]) / data['open_ma%d' % window]
            data['high_ma%d_ratio' % window] = (data['high'] - data['high_ma%d' % window]) / data['high_ma%d' % window]
            data['low_ma%d_ratio' % window] = (data['low'] - data['low_ma%d' % window]) / data['low_ma%d' % window]
            data['close_ma%d_ratio' % window] = (data['close'] - data['close_ma%d' % window]) / data[
                'close_ma%d' % window]
            data['volume_ma%d_ratio' % window] = (data['volume'] - data['volume_ma%d' % window]) / data[
                'volume_ma%d' % window]
        windows1 = [20]
        # BollingerBand
        for window in windows1:
            # 중심선 # 20개 표준편차 # 하한선 # 상한선
            data['close_ma{}'.format(window)] = data['close'].rolling(window).mean()
            data['close_std{}'.format(window)] = data['close'].rolling(window).std()
            data['down_line'] = data['close_ma{}'.format(window)] - data['close_std{}'.format(window)] * 2
            data['upper_line'] = data['close_ma{}'.format(window)] + data['close_std{}'.format(window)] * 2
            data['percent_b'] = (data['close'] - data['down_line']) / (data['upper_line'] - data['down_line'])
            data['Band_width'] = (data['upper_line'] - data['down_line']) / data['close_ma{}'.format(window)]
        # DMI
        data['PDM'] = np.zeros(len(data))
        data.loc[1:, 'PDM'] = np.abs(
            data.loc[1:, 'high'].copy().values - data.loc[:len(data) - 2, 'high'].copy().values)
        data['MDM'] = np.zeros(len(data))
        data.loc[1:, 'MDM'] = np.abs(data.loc[1:, 'low'].copy().values - data.loc[:len(data) - 2, 'low'].copy().values)
        data['TR'] = np.zeros(len(data))
        data.loc[1:, 'TR'] = np.maximum(np.abs(data.loc[1:, 'high'].copy().values - data.loc[1:, 'low'].copy().values),
                                        np.abs(data.loc[1:, 'high'].copy().values - data.loc[:len(data) - 2,
                                                                                    'close'].copy().values),
                                        np.abs(data.loc[1:, 'low'].copy().values - data.loc[:len(data) - 2,
                                                                                   'close'].copy().values))
        windows2 = [10]
        for window in windows2:
            data['PDI'] = data['PDM'].rolling(window).mean() / data['TR'].rolling(window).mean()
            data['MDI'] = data['MDM'].rolling(window).mean() / data['TR'].rolling(window).mean()
        data['ADX'] = np.abs(data['PDI'] - data['MDI']) / (data['PDI'] + data['MDI']) * 100
        # STOCHASTIC
        windows3 = [5]
        for window in windows3:
            data['percent_k'] = (data['close'] - data['close'].rolling(window).min()) / (
                        data['close'].rolling(window).max() - data['close'].rolling(window).min())
        windows4 = [3]
        for window in windows4:
            data['percent_d'] = data['percent_k'].rolling(window).mean()
        # MACD
        data['ema12'] = data['close'].ewm(12).mean()
        data['ema26'] = data['close'].ewm(26).mean()
        data['MACD'] = data['ema12'] - data['ema26']
        data['SIGNAL'] = data['MACD'].ewm(9).mean()
        # RSI
        windows5 = [14]
        data['U'] = np.where(data['close'].diff(1) > 0, data['close'].diff(1), 0)
        data['D'] = np.where(data['close'].diff(1) < 0, data['close'].diff(1) * (-1), 0)
        for window in windows5:
            data['RSI'] = data['U'].rolling(window).mean() / (
                        data['U'].rolling(window).mean() + data['D'].rolling(window).mean())
        # Momentum
        data['Momentum'] = np.zeros(len(data))
        data.loc[10:, 'Momentum'] = (data.loc[10:, 'close'].copy().values / data.loc[:len(data) - 11,
                                                                            'close'].copy().values) * 100

        # 주가, 거래량 전 대비 비율
        data['open_lastopen_ratio'] = np.zeros(len(data))
        data.loc[1:, 'open_lastopen_ratio'] = (data.loc[1:, 'open'].copy().values - data.loc[:len(data) - 2,
                                                                                    'open'].copy().values) / data.loc[
                                                                                                             :len(
                                                                                                                 data) - 2,
                                                                                                             'open'].copy().values
        data['high_lasthigh_ratio'] = np.zeros(len(data))
        data.loc[1:, 'high_lasthigh_ratio'] = (data.loc[1:, 'high'].copy().values - data.loc[:len(data) - 2,
                                                                                    'high'].copy().values) / data.loc[
                                                                                                             :len(
                                                                                                                 data) - 2,
                                                                                                             'high'].copy().values
        data['low_lastlow_ratio'] = np.zeros(len(data))
        data.loc[1:, 'low_lastlow_ratio'] = (data.loc[1:, 'low'].copy().values - data.loc[:len(data) - 2,
                                                                                 'low'].copy().values) / data.loc[
                                                                                                         :len(data) - 2,
                                                                                                         'low'].copy().values
        data['close_lastclose_ratio'] = np.zeros(len(data))
        data.loc[1:, 'close_lastclose_ratio'] = (data.loc[1:, 'close'].copy().values - data.loc[:len(data) - 2,
                                                                                       'close'].copy().values) / data.loc[
                                                                                                                 :len(
                                                                                                                     data) - 2,
                                                                                                                 'close'].copy().values
        data['volume_lastvolume_ratio'] = np.zeros(len(data))
        data.loc[1:, 'volume_lastvolume_ratio'] = (data.loc[1:, 'volume'].copy().values - data.loc[:len(data) - 2,
                                                                                          'volume'].copy().values) / data.loc[
                                                                                                                     :len(
                                                                                                                         data) - 2,
                                                                                                                     'volume'].copy().replace(
            to_replace=0, method='ffill').replace(to_replace=0, method='bfill').values
        data['next_close_ratio'] = np.zeros(len(data))
        data.loc[:len(data) - 2, 'next_close_ratio'] = (data.loc[1:, 'close'].copy().values - data.loc[:len(data) - 2,
                                                                                              'close'].copy().values) / data.loc[
                                                                                                                        :len(
                                                                                                                            data) - 2,
                                                                                                                        'close'].copy().values * 100
    return data

def load_data(fpath, ver, start_time=None, end_time=None):
    if ver == 'ossp_all':
        data = pd.read_csv(fpath, sep='\t', names=COLUMNS_CHART_DATA_N, header=None)
    if ver == 'ossp_lstm':
        data = pd.read_csv(fpath, sep='\t', names=COLUMNS_CHART_DATA_N, header=None)
    data = data.sort_values(by='time')
    data.reset_index(drop=True, inplace=True)
    # 데이터 전처리
    data = preprocess(data, ver=ver)
    #    data['date'] = data['date'].str.replace('-', '')
    # 필터링한 기간에 따른 index조절
    # 이렇게 하면 분봉조회로 가져온 데이터는 역순으로 담겨있기 때문에
    # 시간순으로 다시 재배치(크기순 배치)
    # 결측값 제거 - rolling(window)연산 시 window만큼의 행은 NaN으로 나타남(자동삭제 가능)
    data = data[(data['time'] >= int(start_time)) & (data['time'] <= int(end_time))]
    data = data.dropna()

    # 차트 데이터 분리
    if ver == 'ossp_all':
        #chart_data = data[COLUMNS_CHART_DATA_OSSP_ALL]
        chart_data = data[COLUMNS_CHART_DATA_N]
    else:
        chart_data = data[COLUMNS_CHART_DATA_K]
    # 학습 데이터 분리
    training_data = None
    if ver == 'ossp_all':
        training_data = data[COLUMNS_TRAINING_DATA_OSSP_ALL]
    ##################
    elif ver == 'ossp_lstm':
        pass
    else:
        raise Exception('Invalid version.')
    #print(chart_data)
    #print(training_data)

    if ver == 'ossp_lstm':
        chart_data = None
        training_data = None
        chart_data = data[COLUMNS_CHART_DATA_OSSP_ALL]
        training_data = data[COLUMNS_TRAINING_DATA_OSSP_ALL]
        training_data, chart_data = make_dataset(training_data, chart_data, 5)

    chart_data = chart_data.astype(np.float32)
    training_data = training_data.astype(np.float32)
    return chart_data, training_data