# coding=utf-8
import argparse, os
from rl.policy import LinearAnnealedPolicy,BoltzmannQPolicy,EpsGreedyQPolicy #,SoftmaxPolicy
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
# return [memory] & associated [policy] information
def getAgentFactors(MEMORY_LIMIT,VALUE_TEST):
    # define associated memories
    memory=SequentialMemory(limit=MEMORY_LIMIT,window_length=1)
    # define associated processor
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=.995, value_min=.005, value_test=VALUE_TEST,nb_steps=10000)

    # return associated parts
    return memory,policy


# return the associated agent information
def buildDQN(model,num_classes,policy,memory,processor,GAMMA,BATCH_SIZE):

    dqn = DQNAgent(model=model,
                   nb_actions=num_classes,
                   policy=policy,
                   memory=memory,
                   nb_steps_warmup=50000,
                   gamma=GAMMA,
                   target_model_update=0.6,
                   batch_size=BATCH_SIZE,
                   train_interval=1,
                   delta_clip=0.7,
                   processor=processor,
                   enable_double_dqn=False,
                   enable_dueling_network=False)

    dqn.compile(Adam(lr=.0001), metrics=['mse','categorical_crossentropy']) #,'mae'

    return dqn

# update the agent with [trained weights]
def readyForTest(dqn,checkpoint_path):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    dqn.load_weights(latest)

    return dqn
