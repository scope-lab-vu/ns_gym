import json
import logging
import os
from kafka import KafkaProducer, KafkaConsumer, TopicPartition
from kafka_utils import model_weights_to_str
from ns_gym.benchmark_algorithms.AlphaZero.alphazero import train_model

logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S', level=logging.CRITICAL,
                    handlers=[logging.FileHandler(f"logs/debug_training_process_{os.getpid()}.log", mode='w')])
logger = logging.getLogger()







def nn_training(args_dict):
    """
    train the neural network
    pre: args_dict contains parameters configuration
    post: send the trained neural network
    """
    message_list = []
    replay_buffer = []
    weights_index = 0
    # unzip parameters from args_dict
    train_from_start = args_dict["train_from_start"]
    MSG_LEN = args_dict["MSG_LEN"]
    logging_level = args_dict["logging_level"]
    # saved network weights locally as backup weights
    Network_Weights = args_dict["Network_Weights"]
    weights_topic = args_dict["weights_topic"]
    data_topic = args_dict["data_topic"]
    num_hidden_layers = args_dict["num_hidden_layers"]
    outer_training_times = args_dict["outer_training_times"]
    inner_training_times = args_dict["inner_training_times"]
    batch_size = args_dict["batch_size"]
    epochs = args_dict["epochs"]
    # logger = args_dict["logger"]
    logger.critical(f"start training process {os.getpid()}")
    counter = 0
    trash_list = []  # show training data that throw away, debug use

    bootstrap_server_name = "localhost:9092"

    # kafka producers and consumers
    producer = KafkaProducer(bootstrap_servers=[bootstrap_server_name], retries=1)
    consumer = KafkaConsumer(
        bootstrap_servers=[bootstrap_server_name],
        auto_offset_reset='latest',
        enable_auto_commit=True
    )

    # kafka topics
    topic = TopicPartition(topic=data_topic, partition=0)
    consumer.assign([topic])

    # load neural network
    if train_from_start:
        network = build_model(num_hidden_layers=num_hidden_layers)
        weight_strings = model_weights_to_str(network)
        producer.send(weights_topic, bytes(json.dumps({weights_index: weight_strings}), 'utf-8'))
    else:
        network = build_model(num_hidden_layers=num_hidden_layers)
        network.load_weights(Network_Weights)
        weight_strings = model_weights_to_str(network)
        producer.send(weights_topic, bytes(json.dumps({weights_index: weight_strings}), 'utf-8'))

    # infinite while loop
    while True:
        message = consumer.poll()
        if message:
            message = message[topic]

        if counter >= outer_training_times:
            logger.critical("start get training data")
            weights_index += 1
            weight_strings = model_weights_to_str(network)
            producer.send(weights_topic, bytes(json.dumps({weights_index: weight_strings}), 'utf-8'))
            counter = 0
            # save the weights
            network.save_weights(Network_Weights) #change this

        if len(message_list) >= MSG_LEN:
            training_data_seq = []
            for m in message_list:
                weights_index_m = int(list(m.keys())[0])
                training_data = m[str(weights_index_m)]
                training_data = json.loads(training_data)
                training_data_seq.append(training_data)
            replay_buffer.extend(training_data_seq)
            message_list = []
            replay_buffer = replay_buffer[-10000:]
            logger.critical("before calling train model")
            network = train_model(network, replay_buffer, MSG_LEN, inner_training_times, batch_size, epochs)
            logger.critical("after calling train model")
            counter += 1
        else:
            for m in message:
                # m is a dictionary, check m's key
                m = m.value.decode('utf-8')
                # m = str_to_dict(m)
                m = json.loads(m)
                if int(list(m.keys())[0]) == weights_index:
                    message_list.append(m)
                # else:
                #     trash_list.append(m)
                #     if len(trash_list) > 10:
                #         print(f"trash_list: {trash_list}")
