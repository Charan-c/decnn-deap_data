# decnn-deap_data
# Emotion Recognition from EEG Signals using Dynamic Differential Entropy and CNN

This repository contains the code and resources for an emotion recognition model based on Dynamic Differential Entropy (DDE) and Convolutional Neural Networks (CNN), applied to EEG signals. The aim of the project is to classify emotional states based on EEG data, using a subject-independent approach.

## Dataset

The **DEAP (Database for Emotion Analysis using Physiological Signals)** dataset is used for this project. DEAP contains EEG, peripheral physiological, and audiovisual recordings of participants watching music videos. Each participant rated their emotional states during the recordings, providing valence and arousal labels. The dataset consists of EEG signals from 32 participants watching 40 music videos.

- Dataset: [DEAP Dataset](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/index.html)

  ## Paper

The project is based on the following research paper:

**Title:** Subject-independent Emotion Recognition of EEG Signals Based on Dynamic Empirical Convolutional Neural Network

**Authors:** Shuaiqi Liu, Xu Wang, Ling Zhao, Jie Zhao, Qi Xin, Shui-Hua Wang

**Abstract:** Affective computing is one of the key technologies to achieve advanced brain-machine interfacing. It is increasingly concerning research orientation in the field of artificial intelligence. Emotion recognition is closely related to affective computing. Although emotion recognition based on electroencephalogram (EEG) has attracted more and more attention at home and abroad, subject-independent emotion recognition still faces enormous challenges. We proposed a subject-independent emotion recognition algorithm based on dynamic empirical convolutional neural network (DECNN) in view of the challenges. Combining the advantages of **empirical mode decomposition** (EMD) and **differential entropy** (DE), we proposed a dynamic differential entropy (DDE) algorithm to extract the features of EEG signals. After that, the extracted DDE features were classified by convolutional neural networks (CNN). Finally, the proposed algorithm is verified on SJTU Emotion EEG Dataset (SEED). In addition, we discuss the brain area closely related to emotion and design the best profile of electrode placements to reduce the calculation and complexity. Experimental results show that the accuracy of this algorithm is 3.53% higher than that of the state-of-the-art emotion recognition methods. What’s more, we studied the key electrodes for EEG emotion recognition, which is of guiding significance for the development of wearable EEG devices.

## Block Diagram
[Block Diagram.pdf](https://github.com/Charan-c/decnn-deap_data/files/12288035/Block.Diagram.pdf)

