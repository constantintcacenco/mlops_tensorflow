import tensorflow as tf
import tf2onnx
import os


with tf.compat.v1.Session() as sess:
    x = tf.compat.v1.placeholder(tf.float32, [2, 3], name="input")
    x_ = tf.add(x, x)
    _ = tf.identity(x_, name="output")
    onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph, input_names=["input:0"], output_names=["output:0"])
    model_proto = onnx_graph.make_model("test")
    
    output_dir = os.path.join("outputs","models")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir,"tensorflow_mnist.onnx")

    with open(model_path, "wb") as f:
        f.write(model_proto.SerializeToString())