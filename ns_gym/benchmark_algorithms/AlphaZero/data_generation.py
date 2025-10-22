import logging
import gym
import os
import multiprocessing
import json
from kafka import KafkaProducer, KafkaConsumer
from kafka.structs import TopicPartition
from kafka_utils import str_to_model_weights
import warnings
import gc

warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO,
                    handlers=[logging.FileHandler(f"logs/debug_generator_process_{os.getpid()}", mode='w')])
logger = logging.getLogger()


def data_generation(args_dict):
    # read parpameters
    weights_topic = args_dict["weights_topic"]
    data_topic = args_dict["data_topic"]
    cumulative_reward_topic = args_dict["cumulative_reward_topic"]
    map_name = args_dict["map_name"]
    is_slippery = args_dict["is_slippery"]
    num_iterations = args_dict["num_iterations"]
    num_hidden_layers = args_dict["num_hidden_layers"]
    c_puct = args_dict["c_puct"]
    gamma = args_dict["gamma"]
    logging_level = args_dict["logging_level"]
    # logger = args_dict["logger"]
    network_loaded = True
    weights_index = 0
    num_episodes = 50
    network = build_model(num_hidden_layers)

    bootstrap_server_name = "localhost:9092"

    producer = KafkaProducer(bootstrap_servers=[bootstrap_server_name])
    consumer = KafkaConsumer(
        group_id='consumer2',
        bootstrap_servers=[bootstrap_server_name],
        auto_offset_reset='latest',
        enable_auto_commit=True
    )
    tp = TopicPartition(topic=weights_topic, partition=0)

    consumer.assign([tp])

    while True:
        try:
            logger.critical("begin episode")
            env = gym.make('FrozenLake-v1', map_name=map_name, is_slippery=is_slippery) #TODO: change to my environment

            lock = multiprocessing.Lock()
            lock.acquire()
            # network = build_model(num_hidden_layers)
            message = consumer.poll(max_records=1)
            # message = consumer.poll()
            if message: 
                message = message[tp]
                # msg_dict = json.loads(message[tp][0].value.decode('utf-8'))
                msg_dict = json.loads(message[0].value.decode('utf-8'))
                weights_index = list(msg_dict.keys())[0]
                weights = msg_dict[weights_index]
                network = str_to_model_weights(network, weights)
                network_loaded = True
            lock.release()

            if network_loaded:
                curr_state = env.reset()
                search_agent = MCTS(current_state=curr_state, gamma=gamma, num_iter=num_iterations, c_puct=c_puct)
                terminated = False
                counter = 0
                training_data_seq = []
                cumulative_reward = 0

                while not terminated:
                    action = search_agent.run_mcts(network=network, env=env)
                    tmpresult = search_agent.get_training_data()
                    # training_data_seq.append(tmpresult)
                    trainingrow = json.dumps(tmpresult)
                    producer.send(data_topic, bytes(json.dumps({weights_index: trainingrow}), 'utf-8'), partition=0)
                    # producer.send(data_topic, bytes(json.dumps({0: json.dumps([0, [0.7, 0.1, 0.1, 0.1], 0.5])}), 'utf-8'), partition=0)

                    # action = env.action_space.sample()
                    curr_state, reward, terminated, _ = env.step(action)
                    cumulative_reward += reward
                    # search_agent.mcts_update_root(curr_state)
                    # comment later, just debug use for print
                    print(os.getpid(), counter, curr_state, reward, terminated, action)
                    counter += 1

                    # if terminated:
                    #     producer.send(data_topic, bytes(json.dumps({weights_index: trainingrow}), 'utf-8'), partition=0)

                tf.keras.backend.clear_session()
                _ = gc.collect()

                producer.send(cumulative_reward_topic, bytes(json.dumps(cumulative_reward), 'utf-8'), partition=0)

        except:
            logging.critical("Exception occurred")
            traceback.print_exc()
