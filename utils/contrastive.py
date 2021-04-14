import tensorflow as tf
import tensorflow_addons as tfa

LARGE_NUM = 1e9


def compute_logits_and_labels(hidden, temperature=0.5, hidden_norm=True):
    if hidden_norm:
        hidden = tf.math.l2_normalize(hidden, -1)

    hidden1, hidden2 = tf.split(hidden, 2, 0)
    batch_size = tf.shape(hidden1)[0]
    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

    logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
    logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature
    return logits_ab, logits_aa, logits_ba, logits_bb, labels


class NTXentLoss(tf.keras.losses.Loss):

    def __init__(self, hidden_norm=True, temperature=1.0, *args, **kwargs):
        super().__init__(name='con_loss')
        self.hidden_norm = hidden_norm
        self.temperature = temperature
        self.cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    def call(self, y_true, hidden):
        logits_ab, logits_aa, logits_ba, logits_bb, labels = compute_logits_and_labels(hidden,
                                                                                       temperature=self.temperature,
                                                                                       hidden_norm=self.hidden_norm)
        loss_a = self.cross_entropy_loss(labels, tf.concat([logits_ab, logits_aa], 1))
        loss_b = self.cross_entropy_loss(labels, tf.concat([logits_ba, logits_bb], 1))
        loss = loss_a + loss_b
        return loss


def contrastive_acc_fn(temperature=0.5, hidden_norm=True):
    def contrastive_acc(y_true, hidden):
        logits_ab, logits_aa, logits_ba, logits_bb, labels = compute_logits_and_labels(hidden,
                                                                                       temperature=temperature,
                                                                                       hidden_norm=hidden_norm)
        return tf.equal(tf.argmax(labels, 1), tf.argmax(logits_ab, axis=1))

    return contrastive_acc


class ContrastiveAccuracy(tfa.metrics.MeanMetricWrapper):

    def __init__(self, name='contrastive_acc', dtype=None, temperature=0.5, hidden_norm=True):
        super(ContrastiveAccuracy, self).__init__(contrastive_acc_fn(temperature=temperature, hidden_norm=hidden_norm),
                                                  name, dtype=dtype)


def contrastive_entropy_fn(temperature=0.5, hidden_norm=True):
    def contrastive_entropy(y_true, hidden):
        logits_ab, logits_aa, logits_ba, logits_bb, labels = compute_logits_and_labels(hidden,
                                                                                       temperature=temperature,
                                                                                       hidden_norm=hidden_norm)
        prob_con = tf.nn.softmax(logits_ab)
        entropy_con = - tf.reduce_mean(
            tf.reduce_sum(prob_con * tf.math.log(prob_con + 1e-8), -1))
        return entropy_con

    return contrastive_entropy


class ContrastiveEntropy(tfa.metrics.MeanMetricWrapper):

    def __init__(self, name='contrastive_entropy', dtype=None, temperature=0.5, hidden_norm=True):
        super(ContrastiveEntropy, self).__init__(contrastive_entropy_fn(temperature=temperature, hidden_norm=hidden_norm),
                                                 name, dtype=dtype)
