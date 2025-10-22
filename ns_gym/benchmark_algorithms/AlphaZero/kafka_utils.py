import json
import numpy as np
from kafka import KafkaAdminClient


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def model_weights_to_str(model):
    d = []
    for i, l in enumerate(model.layers):
        w = l.get_weights()
        d.append(w)
    json_dump = json.dumps(d, cls=NumpyEncoder)
    return json_dump


def str_to_model_weights(model, json_dumps):
    json_load = json.loads(json_dumps)
    for i, l in enumerate(model.layers):
        w = json_load[i]
        new_w = []
        for _w in w:
            w = np.array(_w)
            new_w.append(w)
        model.layers[i].set_weights(new_w)
    return model


def delete_topics(topic_names):
    admin_client = KafkaAdminClient(bootstrap_servers=['hyper12.isis.vanderbilt.edu:9093'])
    try:
        admin_client.delete_topics(topics=topic_names)
        print("Topic Deleted Successfully")
    except Exception:
        print("Topic Doesn't Exist")


def str_to_dict(str_dict):
    """
    Convert a string representation of a dictionary to a dictionary
    """
    str_dict = str_dict.strip("{}")
    pairs = str_dict.split(":")
    key = int(pairs[0])
    value = pairs[1].strip()
    dict = {key: value}
    return dict
