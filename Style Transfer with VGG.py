import numpy as np
import time
import tensorflow as tf
import vgg
import os
from PIL import Image

# print(os.getcwd())

# Images
ImgOutName = 'FilterBild_v2.'
# Achtung : immer mit / am ende
output_path = 'C:/Users/thi/Desktop/Transfer Test 3/'
# Styleimage wich will be train on
style_path = 'C:/Users/thi/Desktop/style transfer images/style images 256x256 png/Gletscher.png'
# Content image wich is used
content_path = 'C:/Users/thi/Desktop/style transfer images/conten images 256x256 png/Landschaft.png'

# Parameter bzw Hyperparameter
# weights
content_weight = 0  # Gewichtung des "Contentloss" [stand. 5e0]
style_weight = 10  # Gewichtng des "Styleloss" [stand. 1e2]
tv_weight = 1  # Gewichtung des "total variation loss" [stand. 1e-5]

# Training
learning_rate = 10.  # Learning rate [stand. 10.]
num_iterations = 1001  # Iterationen [stand. 1000]

# VGG Netzwerk
# ort der VGG "weights"
vgg_path = "real-time style transfer/imagenet-vgg-verydeep-19.mat"
# Layers welche vo VGG Net genommen werden
content_layers = "relu4_4"
style_layers = "relu1_2,relu2_2,relu3_4,relu4_4"

# Wenn True startete das mixed Image mit zufälligen Pixeln
random_init = True

# erstellen von Listen aus Inputstring
style_layers = style_layers.split(',')
content_layers = content_layers.split(',')


def get_image(GetPath):
    return tf.expand_dims((np.array(Image.open(GetPath))).astype('float32'), axis=0)


def output_image(tensor, SavePath, name):
    imgOut = tf.image.encode_png(tf.cast(tf.squeeze(tensor), tf.uint8)) #squeeze um die Batch-dimension des Tensors zu löschen
    return tf.write_file((SavePath + name + '.png'),imgOut)


# Errorberechnungsfunktion
def mean_squared_error(a, b):
    return tf.reduce_mean(tf.square(a - b))

# loss Funktionen
# denoise loss
def total_variation_loss(layer):
    return tf.reduce_sum(
        tf.image.total_variation(layer, name='total_variation_loss_function'))


# gram Funktion für Style loss
def gram(layer):
    shape=tf.shape(layer)
    num_filters=shape[3]
    matrix=tf.reshape(layer, shape=[-1, num_filters])
    # tf.transpose "dreht die matrix um 90"
    return tf.matmul(tf.transpose(matrix), matrix)

#  VGG weight/features import
def get_style_features(style_path, style_layers):
    with tf.Graph().as_default() as g:
        image=get_image(style_path)
        net, _=vgg.net(vgg_path, image)
        features=[]
        for layer in style_layers:
            features.append(gram(net[layer]))

        with tf.Session() as sess:
            return sess.run(features)

#  VGG weight/features import
def get_content_features(content_path, content_layers):
    with tf.Graph().as_default() as g:
        image=get_image(content_path)
        net, _=vgg.net(vgg_path, image)
        layers=[]
        for layer in content_layers:
            layers.append(net[layer])

        with tf.Session() as sess:
            return sess.run(layers + [image])


def main(argv=None):


    style_features_t=get_style_features(style_path, style_layers)
    res=get_content_features(content_path, content_layers)
    # array wird in bild und features aufgeteilt
    content_features_t, image_t=res[:-1], res[-1]

    image=tf.constant(image_t)
    random=tf.random_normal(tf.shape(image_t))
    initial=tf.Variable(random if random_init else image)

    net, _=vgg.net(vgg_path, initial)

    content_loss=0
    for content_features, layer in zip(content_features_t, content_layers):
        content_loss += mean_squared_error(content_features, net[layer])
    content_loss=content_weight * content_loss / len(content_layers)

    style_loss=0
    for style_gram, layer in zip(style_features_t, style_layers):
        style_loss += mean_squared_error(gram(net[layer]), style_gram)
    style_loss=style_weight * style_loss

    tv_loss=tv_weight * total_variation_loss(initial)

    total_loss=content_loss + style_loss + tv_loss #addieren der Losswerte

    train_op=tf.train.AdamOptimizer(learning_rate).minimize(total_loss) #Implementation von AdamOptimizer

# Training wird gestartet
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        start_time=time.time()
        for step in range(num_iterations):
            _, loss_t, cl, sl=sess.run(
                [train_op, total_loss, content_loss, style_loss])
            elapsed=time.time() - start_time
            start_time=time.time()
            print(step, ';', elapsed, ';', loss_t, ';', cl, ';', sl)
            if step % 50 == 0:
                sess.run(output_image(initial, output_path, ImgOutName + str(step)))


if __name__ == '__main__':
    main()
