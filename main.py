import os
import sys
import logging
import argparse
import json

import settings
import utils
import data_manager

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock_code', nargs='+')
    parser.add_argument('--ver', choices=['al', 'v1', 'v2', 'ossp_all'], default='a1')
    parser.add_argument('--rl_method',
                        choices=['dqn', 'pg', 'ac', 'a2c', 'a3c'], default='a3c')
    parser.add_argument('--net',
                        choices=['dnn', 'lstm', 'cnn'], default='lstm')
    parser.add_argument('--num_steps', type=int, default=5) # dnn제외 lstm, cnn을 사용할 경우에만 사용
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--discount_factor', type=float, default=0.9)
    parser.add_argument('--start_epsilon', type=float, default=1)
    parser.add_argument('--balance', type=int, default=10000000)
    parser.add_argument('--num_epoches', type=int, default=100)
    parser.add_argument('--delayed_reward_threshold',
                        type=float, default=0.1) #1분봉에서는 다 거기서 거기이기 때문에 한 번의 거래에 지연 보상 발생 X 아예 거래세, 수수료 0.00265보다 조금만 높아도 지연 보상으로 지칭
    parser.add_argument('--backend',
                        choices=['tensorflow', 'plaidml'], default='tensorflow')
    parser.add_argument('--output_name', default=utils.get_time_str())
    parser.add_argument('--value_network_name')
    parser.add_argument('--policy_network_name')
    parser.add_argument('--reuse_models', action='store_true')
    parser.add_argument('--learning', action='store_true')
    ####################################################### 20210515
    parser.add_argument('--application_private', action='store_true')
    ####################################################### 20210515
    # 분봉의 경우
    #parser.add_argument('--date', default='20200626') # 투자를 할 일자를 선택 09:10 ~ 14:59 (15:19까지가 정상거래 -> 15:20 ~ 30분은 동시호가 진행이기에 무의미 -> 넉넉히 14:59까지로
    parser.add_argument('--start_date', default='20200626')
    parser.add_argument('--end_date', default='20200626')

    args = parser.parse_args()

    # Keras Backend 설정
    if args.backend == 'tensorflow':
        os.environ['KERAS_BACKEND'] = 'tensorflow'
    elif args.backend == 'plaidml':
        os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

    # 출력 경로 output폴더에 output_name_강화학습기법_신경망구조의 이름 폴더 생성
    output_path = os.path.join(settings.BASE_DIR, 'output/{}_{}_{}'.format(args.output_name, args.rl_method, args.net))
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # 파라미터 기록
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(json.dumps(vars(args)))

    # 로그 기록 설정 위의 출력 경로에 log파일 생성
    file_handler = logging.FileHandler(filename=os.path.join(output_path, "{}.log".format(args.output_name)), encoding='utf-8')
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setLevel(logging.DEBUG)
    stream_handler.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s", handlers=[file_handler, stream_handler], level=logging.DEBUG)

    # 로그, Keras Backend 설정을 먼저하고 RLTrader 모듈들을 이후에 임포트해야 함
    from agent import Agent
    from learners import DQNLearner, PolicyGradientLearner, ActorCriticLearner, A2CLearner, A3CLearner

    # 모델 경로 준비
    value_network_path = ''
    policy_network_path = ''
    ######################### 삭제가능
    if args.value_network_name is not None:
        value_network_path = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(args.value_network_name))
    else:
        value_network_path = os.path.join(output_path, '{}_{}_value_{}.h5'.format(args.rl_method, args.net, args.output_name))
    if args.policy_network_name is not None:
        policy_network_path = os.path.join(settings.BASE_DIR, 'models/{}.h5'.format(args.policy_network_name))
    else:
        policy_network_path = os.path.join(output_path, '{}_{}_policy_{}.h5'.format(args.rl_method, args.net, args.output_name))

    common_params = {}
    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_unit = []
    list_max_trading_unit = []
    # 분봉의 경우
    # start_time = args.date + "090000"  # 9시 10분부터
    # end_time = args.date + "150000"  # 15시 까지
    start_time = args.start_date
    end_time = args.end_date
    for stock_code in args.stock_code:
        #################################################### 20210514 추가본, 각 주식 코드별로 구현
        ####### 예를 들어, reuse_models하고 싶은 애가 있으면, 그 폴더의 이름 + 뒤에 _test 붙이기, 그리고 그 폴더에 신경망 h5 집어넣기 그후
        ####### if(args.reuse_models) 구문에 path에 넣어주기 인자 실행 시
        ####### 각 주식종목 별로 수행(각 주식종목 별 신경망 생성) --application_private
        ####### 주식종목 별이 아닌 공유 신경망 생성(한 신경망으로 계속 학습) (따로 인자 X)
        if (args.application_private):
            output_name = stock_code #stock_code
            if (args.reuse_models):
                value_network_path = os.path.join(settings.BASE_DIR,
                                                  'output/20210601_dqn_dnn_test/{}_{}_value_{}.h5'.format(args.rl_method, args.net, output_name))
                policy_network_path = os.path.join(settings.BASE_DIR,
                                                  'output/20210601_dqn_dnn_test/{}_{}_policy_{}.h5'.format(args.rl_method, args.net, output_name))
            else:
                value_network_path = os.path.join(output_path,
                                                  '{}_{}_value_{}.h5'.format(args.rl_method, args.net, output_name))
                policy_network_path = os.path.join(output_path,
                                                   '{}_{}_policy_{}.h5'.format(args.rl_method, args.net, output_name))
        else:
            output_name = '068270'# args.output_name
            if (args.reuse_models):
                value_network_path = os.path.join(settings.BASE_DIR,
                                                  'output/20210601_dqn_dnn_test/{}_{}_value_{}.h5'.format(
                                                      args.rl_method, args.net, output_name))
                policy_network_path = os.path.join(settings.BASE_DIR,
                                                   'output/20210601_dqn_dnn_test/{}_{}_policy_{}.h5'.format(
                                                       args.rl_method, args.net, output_name))
            else:
                value_network_path = os.path.join(output_path,
                                                  '{}_{}_value_{}.h5'.format(args.rl_method, args.net, output_name))
                policy_network_path = os.path.join(output_path,
                                                   '{}_{}_policy_{}.h5'.format(args.rl_method, args.net, output_name))
        #################################################### 20210514 추가본, 각 주식 코드별로 구현
        # 차트 데이터, 학습 데이터 준비
        #chart_data, training_data = data_manager.load_data(os.path.join(settings.BASE_DIR,'data/{}/{}_data.txt'.format(args.ver, stock_code)), start_time, end_time, ver=args.ver)
        chart_data, training_data = data_manager.load_data(
            os.path.join(settings.BASE_DIR, 'files/OSSP_KOSPI/{}_day_data.txt'.format(stock_code)), ver=args.ver, start_time=start_time, end_time=end_time)
        # 최소/최대 투자 단위 설정
        min_trading_unit = 1#max(int((args.balance)/100 / chart_data.iloc[-1]['close']), 1)
        max_trading_unit = max(int(args.balance / chart_data.iloc[-1]['close']), 1)

        # 공통 파라미터 설정
        common_params = {'rl_method': args.rl_method,
                         'delayed_reward_threshold': args.delayed_reward_threshold,
                         'net': args.net, 'num_steps': args.num_steps, 'lr': args.lr,
                         'output_path': output_path, 'reuse_models': args.reuse_models}

        # 강화학습 시작
        learner = None
        if args.rl_method != 'a3c':
            common_params.update({'stock_code': stock_code,
                                  'chart_data': chart_data,
                                  'training_data': training_data,
                                  'min_trading_unit': min_trading_unit,
                                  'max_trading_unit': max_trading_unit})
            if args.rl_method == 'dqn':
                learner = DQNLearner(**{**common_params,
                                        'value_network_path': value_network_path})
            elif args.rl_method == 'pg':
                learner = PolicyGradientLearner(**{**common_params,
                                                   'policy_network_path': policy_network_path})
            elif args.rl_method == 'ac':
                learner = ActorCriticLearner(**{**common_params,
                                                'value_network_path': value_network_path,
                                                'policy_network_path': policy_network_path})
            elif args.rl_method == 'a2c':
                learner = A2CLearner(**{**common_params,
                                        'value_network_path': value_network_path,
                                        'policy_network_path': policy_network_path})
            if learner is not None:
                learner.run(balance=args.balance,
                            num_epoches=args.num_epoches,
                            discount_factor=args.discount_factor,
                            start_epsilon=args.start_epsilon,
                            learning=args.learning)
                learner.save_models()
        else:
            list_stock_code.append(stock_code)
            list_chart_data.append(chart_data)
            list_training_data.append(training_data)
            list_min_trading_unit.append(min_trading_unit)
            list_max_trading_unit.append(max_trading_unit)

    if args.rl_method == 'a3c':
        learner = A3CLearner(**{
            **common_params,
            'list_stock_code': list_stock_code,
            'list_chart_data': list_chart_data,
            'list_training_data': list_training_data,
            'list_min_trading_unit': list_min_trading_unit,
            'list_max_trading_unit': list_max_trading_unit,
            'value_network_path': value_network_path,
            'policy_network_path': policy_network_path})

        learner.run(balance=args.balance, num_epoches=args.num_epoches,
                    discount_factor=args.discount_factor,
                    start_epsilon=args.start_epsilon,
                    learning=args.learning)
        learner.save_models()