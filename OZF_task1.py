def makepredictions(input_dir,selfie_dir,scan_dir,exc_dir):

    model_path = './model/model.ckpt'
    tf.logging.set_verbosity(tf.logging.ERROR)
    def weight_variable(shape):
        initial = tf.zeros(shape)
        return tf.Variable(initial)
    def bias_variable(shape):
        initial = tf.zeros(shape)
        return tf.Variable(initial)

    gl_imsize = 224

    graph = tf.Graph()
    with graph.as_default():    
        input_depth = 1
        block0_depth = 8
        block1_depth = 8
        block2_depth = 16
        block3_depth = 16
        
        pool4_size = gl_imsize//2**4
        FC1_depth = 128
        FC2_depth = 50
        FC_output_depth = 2
        
        conv0_0w = weight_variable([3,3,1,block0_depth])
        conv0_0b = bias_variable([block0_depth])
        conv0_1w = weight_variable([3,3,block0_depth,block0_depth])
        conv0_1b = bias_variable([block0_depth])
        
        conv1_0w = weight_variable([3,3,block0_depth,block1_depth])
        conv1_0b = bias_variable([block1_depth])
        conv1_1w = weight_variable([3,3,block1_depth,block1_depth])
        conv1_1b = bias_variable([block1_depth])
        
        conv2_0w = weight_variable([3,3,block1_depth,block2_depth])
        conv2_0b = bias_variable([block2_depth])
        conv2_1w = weight_variable([3,3,block2_depth,block2_depth])
        conv2_1b = bias_variable([block2_depth])
        
        conv3_0w = weight_variable([3,3,block3_depth,block3_depth])
        conv3_0b = bias_variable([block3_depth])
        conv3_1w = weight_variable([3,3,block3_depth,block3_depth])
        conv3_1b = bias_variable([block3_depth])

        FC1_w = weight_variable([pool4_size,pool4_size,block2_depth,FC1_depth])
        FC1_b = bias_variable([FC1_depth])
        FC2_w = weight_variable([1,1,FC1_depth,FC2_depth])
        FC2_b = bias_variable([FC2_depth])
        FC_output_w = weight_variable([1,1,FC2_depth,FC_output_depth])
        FC_output_b = bias_variable([FC_output_depth])
        
        tf_image  = tf.placeholder(tf.float32,shape=(1,gl_imsize,gl_imsize,1))
        
        def model_predict(data):
            conv0_0 = tf.nn.conv2d(data,conv0_0w,strides=[1, 1, 1, 1],padding='SAME') + conv0_0b #NHWC num hei wei chan
            conv0_0 = tf.nn.relu(conv0_0)
            conv0_1 = tf.nn.conv2d(conv0_0,conv0_1w,strides=[1, 1, 1, 1],padding='SAME') + conv0_1b
            conv0_1 = tf.nn.relu(conv0_1)
            pool0   = tf.nn.max_pool(conv0_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
            
            conv1_0 = tf.nn.conv2d(pool0,conv1_0w,strides=[1, 1, 1, 1],padding='SAME') + conv1_0b
            conv1_0 = tf.nn.relu(conv1_0)
            conv1_1 = tf.nn.conv2d(conv1_0,conv1_1w,strides=[1, 1, 1, 1],padding='SAME') + conv1_1b
            conv1_1 = tf.nn.relu(conv1_1)
            pool1   = tf.nn.max_pool(conv1_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
            
            conv2_0 = tf.nn.conv2d(pool1,conv2_0w,strides=[1, 1, 1, 1],padding='SAME') + conv2_0b
            conv2_0 = tf.nn.relu(conv2_0)
            conv2_1 = tf.nn.conv2d(conv2_0,conv2_1w,strides=[1, 1, 1, 1],padding='SAME') + conv2_1b
            conv2_1 = tf.nn.relu(conv2_1)
            pool2   = tf.nn.max_pool(conv2_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
            
            conv3_0 = tf.nn.conv2d(pool2,conv3_0w,strides=[1, 1, 1, 1],padding='SAME') + conv3_0b
            conv3_0 = tf.nn.relu(conv3_0)
            conv3_1 = tf.nn.conv2d(conv3_0,conv3_1w,strides=[1, 1, 1, 1],padding='SAME') + conv3_1b
            conv3_1 = tf.nn.relu(conv3_1)
            pool3   = tf.nn.max_pool(conv3_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
            
            FC1 = tf.nn.conv2d(pool3,FC1_w,strides=[1,1,1,1],padding='VALID') + FC1_b
            FC1 = tf.nn.relu(FC1)
            FC2 = tf.nn.conv2d(FC1,FC2_w,strides=[1,1,1,1],padding='VALID') + FC2_b
            FC2 = tf.nn.relu(FC2)
            FC_output = tf.nn.conv2d(FC2,FC_output_w,strides=[1,1,1,1],padding='VALID')+FC_output_b
            return tf.reshape(FC_output,[FC_output.shape[0],FC_output.shape[3]])
        
        image_prediction = tf.nn.softmax(model_predict(tf_image))
        
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as session:
        saver.restore(session,model_path)
    ##    pre_imsize = 324
    ##    shift = (pre_imsize-gl_imsize)//2
        for file in os.listdir(input_dir):
            try:
                img = cv2.imread(input_dir+'/'+file,0)
                if len(img.shape) != 2:
                    raise Exception
              ##  img = cv2.resize(img,(pre_imsize,pre_imsize),cv2.INTER_LINEAR)
              ##  img = img[shift:pre_imsize-shift,shift:pre_imsize-shift]
                img = cv2.resize(img,(gl_imsize,gl_imsize),cv2.INTER_LINEAR)
                img = img.reshape((img.shape[0],img.shape[1],1))
                prediction = image_prediction.eval({tf_image : [img]})[0]
                prediction = np.argmax(prediction)
                if prediction:
                    cv2.imwrite(selfie_dir+'/'+file,img)
                else:
                    cv2.imwrite(scan_dir+'/'+file,img)
            except Exception:
                copyfile(input_dir+'/'+file,exc_dir+'/'+file)


def main(args):
    input_dir = args[0]
    selfie_dir = args[1]
    scan_dir = args[2]
    exc_dir = args[3]
    makepredictions(input_dir,selfie_dir,scan_dir,exc_dir)

if __name__ == '__main__':
    import sys
    import os
    from shutil import copyfile

    import numpy as np
    import cv2
    import tensorflow as tf

    main(sys.argv[1:])