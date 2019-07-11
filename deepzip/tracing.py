from functools import wraps

import tensorflow as tf

###########################
# Autograph Tracing Utils #
###########################

_TRACE_GRAPHS = True
_TRACE_RECORD = {}


def _add_to_trace_record(func):
    global _TRACE_RECORD
    name = func.__name__
    if name in _TRACE_RECORD:
        count = _TRACE_RECORD[name]
        _TRACE_RECORD[name] += 1
        if count == 10:
            retrace_indicator(name)
    else:
        _TRACE_RECORD[name] = 1


def reset_trace_record():
    global _TRACE_RECORD
    _TRACE_RECORD = {}


def trace_graph(func):
    if _TRACE_GRAPHS:

        @wraps(func)
        @tf.function
        def func_with_trace_count(*args, **kwargs):
            _add_to_trace_record(func)
            return func(*args, **kwargs)

        return func_with_trace_count
    else:
        return func


def is_dynamically_unrolled(f, *args):
    g = f.get_concrete_function(*args).graph
    if any(node.name == "while" for node in g.as_graph_def().node):
        print(
            "{}({}) uses tf.while_loop.".format(f.__name__, ", ".join(map(str, args)))
        )
    elif any(node.name == "ReduceDataset" for node in g.as_graph_def().node):
        print(
            "{}({}) uses tf.data.Dataset.reduce.".format(
                f.__name__, ", ".join(map(str, args))
            )
        )
    else:
        print("{}({}) gets unrolled.".format(f.__name__, ", ".join(map(str, args))))


def retrace_indicator(func_name):
    print(
        f"\nWarning: {func_name} is being traced repeatedly. See "
        "https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/function "
        "for more information."
    )
