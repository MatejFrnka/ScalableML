# **<p style="text-align: center;">Swedish Text Transcription using Transformers</p>**
## **<p style="text-align: center;">ID2223 Lab 2</p>**
## **<p style="text-align: center;">HT 2022</p>**  



<br/><br/>
<br/><br/>
## <p style="text-align: center;">Authors: **Daniel Workinn & Matej Frnka**</p> 
## <p style="text-align: center;">December 2022</p> 

<br/><br/>
<br/><br/>
<br/><br/> 

## 1. Introduction

In this lab, we built a (Swedish) speech to AI generated picture service and made it available online using Huggingface (can be found here: https://huggingface.co/spaces/frnka/swedish-text-to-image). This was done by:

* Downloaded a pretrained automated speech recognition (ASR) model **Whisper**
* Fine-tuned the Whisper model on Swedish using the Common Voice dataset
* Used the model to transcribe the received voice audio to text
* Translated to english using a deepl translator api
* Fed the text into OpenAI's text to image generator DALL-E
* Hosted the service on as an Huggingface app

The full Github repo for this work can be found here: https://github.com/MatejFrnka/ScalableML/tree/master/lab2
<br/><br/> 

## 2. The State of Automated Speech Recognition

In 2020, the state of the art of ASR was greatly improved by unsupervised pre-training techniques as shown by Wav2Vec 2.0. These methods includes pre-training on unlabeled speech datasets, vastly larger than a typical labeled academic speech dataset. The resulting models have very adept audio encoders which learnt high-quality representations of speech. However, due to the data being unlabeled, the decoders are not well trained during this phase. For this reason, fine-tuning with labelled data towards specific tasks is introduced to give the decoder sufficient training. However, fine-tuning can be a complex challenge where performance gains in one ASR task (dataset) does not guarantee good generalization to other ASR tasks (datasets).      

Whisper was developed and released in September 2022 by OpenAI to combat in the shortcomings of the previous models. In contrast to Wav2Vec 2.0, Whisper is pre-trained on labeled data, meaning that both the encoder and decoder are trained during this phase. In turn, this means that the Whisper model can be deploying without further fine-tuning with good performance. This is especially true for English ASR tasks, since the majority of the pre-training dataset is in English. The resulting model showed a stunning improvement in performance, making 55% less errors over a wide range of test datasets compared to Wav2Vec 2.0 as stated in the Whisper paper.
<br/><br/> 

## 3. Improving Performance with Fine-tuning

The performance of the Whisper model can be further improved with fine-tuning of the model. This is especially true for low resource languages - languages with little data in the pre-training dataset. Out of the 680,000 hours of pre-training speech data, only 117,000 hours cover 96 smaller and less frequent languages. And surely, Swedish is one of the low resource languages.  

To fine-tune our model on Swedish, we used the Huggingface Common Voices dataset (sv-SE). This dataset contains 12,360 short audio clips of spoken sentences which amounts to roughly 7-8 hours of data. From this fine-tuning we saw the following performance gains:
* WER: 20.158
* Training loss: 0.00482
<br/><br/> 
## 4. Further Performance Gains

To further improve the performance of the Whisper model, we take a data-centric approach, namely more data is better. As we saw in the previous section, fine-tuning with the Huggingface Common Voices dataset (drastically) improved the WER score of the model when being tested on Swedish language. However, as previously stated the available Swedish speech data in the Huggingface Common voices dataset is limited, only 7-8 hours of fine-tuning data. To the amount of fine-tuning data into context, the movie Titanic contains 2,497 short sentences amounting to 76 minutes of speech in the movie. The Common Voices dataset contains about 5 Titanic movies worth of speech data - not a lot!    

Our suggestion for further improvements of the model is to generate more labeled Swedish speech data to be used for further fine-tuning. The most straight forward way we can see to generate more labeled Swedish speech data is to download Swedish movies including the subtitles. Subtitles are most often delivered as a SRT-file which contains the subtitles of the movie as well as which timestamps the subtitles should be displayed. An extraction of one of these SRT-files is shown below:

```
5
00:02:56,552 --> 00:03:00,055
När jag ser henne träda fram ur mörkret
som ett spökskepp...

6
00:03:00,138 --> 00:03:02,140
blir jag alltid lika berörd.
```

 Given these timestamps, the audio of the movie can be extracted for each timestamp and be mapped to the corresponding subtitle text, thereby generating more labeled speech data which can be used for the model for further fine-tuning.  
 
 We saw X performance gain in Swedish only using the ~7-8 hours of labeled Swedish data in the Huggingface Common Voices dataset. As previously stated, this data amounts to roughly 5 movies worth of data. If we were to generate labeled speech data from 10, or 20, Swedish movies and fine-tune the model further, then surely the performance of Whisper would keep improving for Swedish.



<!-- 1. Describe in your README.md program ways in which you can improve
model performance are using
(a) model-centric approach - e.g., tune hyperparameters, change the
fine-tuning model architecture, etc
(b) data-centric approach - identify new data sources that enable you to
train a better model that one provided in the blog post
If you can show results of improvement, then you get the top grade. -->
