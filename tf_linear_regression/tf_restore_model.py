from sklearn.datasets import fetch_california_housing
import tensorflow as tf

housing = fetch_california_housing()
m, n = housing.data.shape

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "tmp/my_model_final.ckpt")
    best_theta_restored = theta.eval()

    print(best_theta_restored)