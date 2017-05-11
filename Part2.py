import string
import tensorflow as tf
import numpy as np
import copy
from tqdm import tqdm

# We read in the conll file - separating each sentence/entry 
# All lines/words from the same sentence will be in one string
# Returns data: A list where each element is a string containing all lines from the conll file
# that belong to that sentence
def read_conll(file_name):
    with open(file_name) as f:
        data = []
        lines = []
        for line in f.readlines():
            if line == '\n':
                data.append(lines)
                lines = []
            else:
                lines.append(line.lower().split('\t'))
    f.close()
    return data

# create 3 dictionaries - unique words, tags and labels
# include null to use for stacks and buffers with <3 elements
# include unk for unknown words in the test set
def get_dictionaries(train_data):
    w_i = 0
    t_i = 0
    l_i = 0
    
    word_dictionary = {}
    tag_dictionary = {}
    label_dictionary = {}
    
    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            if not train_data[i][j][1] in word_dictionary:
                word_dictionary[train_data[i][j][1]] = w_i
                w_i +=1
            if not train_data[i][j][3] in tag_dictionary:
                tag_dictionary[train_data[i][j][3]] = t_i
                t_i +=1
            if not train_data[i][j][7] in label_dictionary:
                label_dictionary[train_data[i][j][7]] = l_i
                l_i +=1

    word_dictionary['NULL'] = w_i
    tag_dictionary['NULL'] = t_i
    word_dictionary['ROOT'] = w_i + 1
    tag_dictionary['ROOT'] = t_i + 1
    word_dictionary['UNK'] = w_i + 2
    tag_dictionary['UNK'] = t_i + 2
    return word_dictionary, tag_dictionary, label_dictionary

# Create a set of transitions:
# 0 = shift
# (1,i) = right-arc, label
# (-1,i) = left-arc, label
def get_transition_dictionary(label_dictionary):
    counter = 0
    transition_dictionary = {0: counter}
    counter+=1
    for i in range(len(label_dictionary)):
        transition_dictionary[(1,i)] = counter
        transition_dictionary[(-1,i)] = counter + 1
        counter+=2    
    return transition_dictionary

def right_arc(stack, buff, has_no_parent):
    has_no_parent.discard(buff[-1])
    buff[-1] = stack[-1]
    stack.pop()

def left_arc(stack, has_no_parent):
    has_no_parent.discard(stack[-1])
    stack.pop()

def shift(stack, buff):
    stack.append(buff.pop())

# given a sentence and the ids from the top of the stack and buffer:
# return the relationship between them: 0 if they're not related.
# 1 if the stack is the parent, with the label
# -1 if the buffer is the parent, with the label
def getRelation(sentence, stack_top, buff_top):
    if buff_top!=0 and sentence[buff_top - 1][6] == str(stack_top):
        # (parent, child) = right arc (1), label
        return (1, label_dictionary[sentence[buff_top-1][7]])  
    
    elif stack_top!=0 and sentence[stack_top - 1][6] == str(buff_top):
        # (child, parent) = left arc(-1), label index
        return (-1, label_dictionary[sentence[stack_top - 1][7]])
    
    else:
        # no relation = shift
        return (0,-1)

# determine if a word has children by checking if the word is the parent of any word that does not yet have a parent
def hasChildren(sentence, has_no_parent, word_id):
    for i in has_no_parent:
        if sentence[i-1][6] == str(word_id):
            return True
    return False

# format the configuration, returning the top three words of the stack and buffer, and their associated tags (using their index in the global dictionaries)
# if save order is true, also return the local index of the top of the stack and buffer
def format_config(stack,buff,sentence,save_order=False):
    # add top 3 words from buffer and stack or null if len < 3 and the associated tags
    top_stack_words = []
    top_stack_tags = []
    if save_order:
        top_indices = [stack[-1], buff[-1]] if len(stack) > 0 else [1, buff[-1]]
    for i in stack[:-4:-1]:
        if i == 0:
            top_stack_words.append(word_dictionary['ROOT'])
            top_stack_tags.append(tag_dictionary['ROOT'])
        elif (sentence[i-1][1] in word_dictionary) and (sentence[i-1][3] in tag_dictionary):
            top_stack_words.append(word_dictionary[sentence[i-1][1]])
            top_stack_tags.append(tag_dictionary[sentence[i-1][3]])
        elif (sentence[i-1][1] in word_dictionary):
            top_stack_words.append(word_dictionary[sentence[i-1][1]])
            top_stack_tags.append(tag_dictionary['UNK'])
        elif (sentence[i-1][3] in tag_dictionary):  
            top_stack_words.append(word_dictionary['UNK'])
            top_stack_tags.append(tag_dictionary[sentence[i-1][3]])
        else:
            top_stack_words.append(word_dictionary['UNK'])
            top_stack_tags.append(tag_dictionary['UNK'])

    top_stack_words += [word_dictionary['NULL']] * (max(0, (3 - len(stack))))
    top_stack_tags += [tag_dictionary['NULL']] * (max(0, (3 - len(stack))))

    top_buff_words = []
    top_buff_tags = []
    for i in buff[:-4:-1]:
        if i == 0:
            top_buff_words.append(word_dictionary['ROOT'])
            top_buff_tags.append(tag_dictionary['ROOT'])
        elif (sentence[i-1][1] in word_dictionary) and (sentence[i-1][3] in tag_dictionary):
            top_buff_words.append(word_dictionary[sentence[i-1][1]])
            top_buff_tags.append(tag_dictionary[sentence[i-1][3]])
        elif (sentence[i-1][1] in word_dictionary):
            top_buff_words.append(word_dictionary[sentence[i-1][1]])
            top_buff_tags.append(tag_dictionary['UNK'])
        elif (sentence[i-1][3] in tag_dictionary):  
            top_buff_words.append(word_dictionary['UNK'])
            top_buff_tags.append(tag_dictionary[sentence[i-1][3]])
        else:
            top_buff_words.append(word_dictionary['UNK'])
            top_buff_tags.append(tag_dictionary['UNK'])
            

    top_buff_words += [word_dictionary['NULL']] * (max(0, (3 - len(buff))))
    top_buff_tags += [tag_dictionary['NULL']] * (max(0, (3 - len(buff))))
    if save_order:
        return (top_stack_words + top_buff_words), (top_stack_tags + top_buff_tags), top_indices
    else:
        return (top_stack_words + top_buff_words), (top_stack_tags + top_buff_tags)
    
# Algorithm to create training data
def create_config_label_pairs(train_data, save_order = False):            
    train_words = [] 
    train_tags = []
    train_y = [] # transitions
    if save_order:
        top_words_local_indices = [] # holds the local indices of the top of the stack and buff for each config
    
    for sentence in tqdm(train_data): 
        stack = [0] # root
        buff = list(reversed(range(1,len(sentence)+1))) # buff is a list of the word id's going backwards until 1
        has_no_parent = set(range(1, len(sentence)+1))
        
        prev_len = len(train_y)
        while len(buff) != 0:
            # add stack and buffer to train data
            if save_order:
                list_words, list_tags, top_words_loc = format_config(stack, buff, sentence, save_order)
                train_words.append(list_words)
                train_tags.append(list_tags)
                top_words_local_indices.append(top_words_loc)
            else:
                list_words, list_tags = format_config(stack,buff,sentence, save_order)
                train_words.append(list_words)
                train_tags.append(list_tags)               

            # get transition/relation to add as train label and perform the transition:    
            if len(stack) == 0:
                relation = 0
                shift(stack, buff)

            else:
                relation = getRelation(sentence, stack[-1], buff[-1]) # returns tuple (transition type, label)
                if relation[0] == -1 : 
                    left_arc(stack, has_no_parent)
                elif relation[0] == 1 and not hasChildren(sentence, has_no_parent, buff[-1]):
                    right_arc(stack, buff, has_no_parent)
                else:
                    relation = 0
                    shift(stack, buff)

            # add transition to training labels
            train_y.append(transition_dictionary[relation])  
            
    if save_order:
        return train_words, top_words_local_indices, train_tags, train_y
    else:
        return train_words, train_tags, train_y
    
############################## 
# READ IN AND PROCESS THE DATA
##############################
train_data = read_conll('data/train.conll')
dev_data = read_conll('data/dev.conll')
test_data = read_conll('data/test.conll')

word_dictionary, tag_dictionary, label_dictionary = get_dictionaries(train_data)
transition_dictionary = get_transition_dictionary(label_dictionary)

train_words, train_tags, train_y = create_config_label_pairs(train_data)            
dev_words, dev_top_indices, dev_tags, dev_y = create_config_label_pairs(dev_data, save_order = True)            
test_words, test_top_indices, test_tags, test_y = create_config_label_pairs(test_data, save_order = True)            

############################
# Neural Network
############################

# Define the architecture:

# Dimensions of the input embedding
embedding_size = 50

# Dimensions of the hidden embedding
hidden_embedding_size = 200

# Size of the batch to be processed in each step
batch_size = 100

# Number of iterations
num_steps = 594

# Number of epochs
num_epochs = 30

# Number of "sample" classes to pick randomly. 
# sampled_size = 50

# number of unique labels
num_labels = len(transition_dictionary)  

# Length of the words dictionary
num_words = len(word_dictionary) 
# Length of the tags dictionary
num_tags = len(tag_dictionary)

words_in_configuration = 6
tags_in_configuration = 6

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_words = tf.placeholder(tf.int32, shape=[batch_size, words_in_configuration, ])
    tf_train_tags = tf.placeholder(tf.int32, shape=[batch_size, tags_in_configuration, ])
    tf_train_labels = tf.placeholder(tf.int32, shape=[batch_size])
    
    tf_dev_words = tf.constant(dev_words, dtype=tf.int32, shape=[len(dev_words), words_in_configuration, ])
    tf_dev_tags = tf.constant(dev_tags, dtype=tf.int32, shape=[len(dev_tags), tags_in_configuration, ])
    
    tf_test_words = tf.constant(test_words, dtype=tf.int32, shape=[len(test_words), words_in_configuration, ])
    tf_test_tags = tf.constant(test_tags, dtype=tf.int32, shape=[len(test_tags), tags_in_configuration, ])
    # tf_valid_dataset = tf.constant(valid_dataset[50:60])
    # tf_test_dataset = tf.constant(test_dataset)

    # TODO: Generate values with a -1 to 1 range
    tf_word_embedding = tf.Variable(tf.random_uniform([num_words, embedding_size], -1.0, 1.0))
    tf_tag_embedding = tf.Variable(tf.random_uniform([num_tags, embedding_size], -1.0, 1.0))

    tf_weights_words1 = tf.Variable(tf.truncated_normal([words_in_configuration * embedding_size, hidden_embedding_size]))
    tf_weights_tags1 = tf.Variable(tf.truncated_normal([tags_in_configuration * embedding_size, hidden_embedding_size]))
    tf_bias1 = tf.Variable(tf.zeros([hidden_embedding_size]))
    
    tf_weights2 = tf.Variable(tf.truncated_normal([hidden_embedding_size, num_labels]))
    
    def model(input_size, train_words, train_tags):
        # We are going to compute a one hot encoding vector of a very large dataset. We save time of
        # unnecesary computation of the product of the vector with almost all zeros and a matrix and
        # just get the matrix value.
        #tf.slice/tf.concat
        embed_words = tf.reshape(tf.nn.embedding_lookup(tf_word_embedding, train_words), [input_size, words_in_configuration * embedding_size])
        embed_tags = tf.reshape(tf.nn.embedding_lookup(tf_tag_embedding, train_tags), [input_size, tags_in_configuration * embedding_size])

        hidden = tf.pow(tf.matmul(embed_tags, tf_weights_tags1) + tf.matmul(embed_words, tf_weights_words1) + tf_bias1, 3)
        #tf_bias2 = tf.Variable(tf.zeros([num_labels]))

        logits = tf.matmul(hidden, tf_weights2) #+ tf_bias2
        return logits
    
    logits = model(batch_size, tf_train_words, tf_train_tags)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(tf_train_labels, num_labels)))
    
    #optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    optimizer = tf.train.AdagradOptimizer(0.01).minimize(loss)
    # optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss, global_step=global_step)

    prediction = tf.nn.softmax(logits)
    
    # Run similarity for the validation dataset
    valid_prediction = tf.nn.softmax(model(len(dev_words), tf_dev_words, tf_dev_tags))
    test_prediction = tf.nn.softmax(model(len(test_words), tf_test_words, tf_test_tags))
    
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == labels) / predictions.shape[0])

# Run the network:

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    print("Initialized")
    for epoch in range(num_epochs):
        print("Epoch: %d" % epoch)
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (len(train_y) - batch_size)

            batch_words = train_words[offset:(offset + batch_size)]
            batch_tags = train_tags[offset:(offset + batch_size)]
            batch_y = train_y[offset:(offset + batch_size)]
            feed_dict = {tf_train_words: batch_words, tf_train_tags: batch_tags, tf_train_labels: batch_y}
            _, l, predictions = sess.run([optimizer, loss, prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_y))
                #feed_dict = {tf_train_words: dev_w, tf_train_tags: dev_t}
                #valid_predictions = sess.run([prediction], feed_dict=feed_dict)
                print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), dev_y))

    #feed_dict = {tf_train_words: test_w, tf_train_tags: test_t}
    #test_pred = sess.run([prediction], feed_dict=feed_dict)
    print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_y))
    
# Inverse dictionaries
transition_dictionary_inv = {v: k for k, v in transition_dictionary.iteritems()}
label_dictionary_inv = {v: k for k, v in label_dictionary.iteritems()}
word_dictionary_inv = {v: k for k, v in word_dictionary.iteritems()}

# Generate output file - not working
# def generate_output(data, words, tags, topids, y):
#     output = []
#     y_index = 0
#     for s in range(len(data)):
#         sentence = data[s]
#         num_words = len(sentence)
#         temp = copy.copy(sentence)
#         end_sent = False
#         while end_sent == False: 
#             if y[y_index] > 0: # not shift
#                 transition = transition_dictionary_inv[y[y_index]]
#                 if transition[0] == 1:
#                      # right arc
#                     if (topids[y_index][1]-1) < len(sentence):
#                         temp[topids[y_index][1]-1][6] = topids[y_index][0]
#                         temp[topids[y_index][1]-1][7] = label_dictionary_inv[transition[1]]
#                     else:
#                         print "fail"
                
#                 else:
#                     # left arc - same but switch child and parent
#                     if(topids[y_index][0]-1) < len(sentence):
#                         temp[topids[y_index][0]-1][6] = topids[y_index][1]
#                         temp[topids[y_index][0]-1][7] = label_dictionary_inv[transition[1]]
#                     else:
#                         print "fail"
#             if words[y_index] == [3905, 3905, 3905, 3906, 3905, 3905]:
#                 end_sent = True                         
#             y_index +=1
#             if y_index >= len(y):
#                 print y_index
#                 print s
#         output.append(temp)
#     return output