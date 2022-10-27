# D4_baseline
This repository contains the full pipeline to train and evaluate the baseline models in the paper [D4: a Chinese Dialogue Dataset for Depression-Diagnosis-Oriented Chat](https://arxiv.org/abs/2205.11764). To get access to D4, please visit the website of [D4](https://x-lance.github.io/D4/).
## Latest Experiment Results
We split the entire depression diagnosis dialogue procedure into 4 subtasks: 
- **Response Generation** aims to generate doctors' probable responses based on the dialog context; 
- **Topic Prediction** predicts the topic of the response based on the dialogue context. 
- **Dialogue Summary** generates symptom summaries based on the entire dialog history; 
- **Severity Classification** separately predicts the severity of depressive episodes and the suicide risk based on the dialogue context and dialogue summary. 

### Response Generation and Topic Prediction

In our experiments, we jointly optimize the topic prediction model and the response generation model. We take the topic as a special first token of dialogue response.

*− means topics are excluded, BERT means topics predicted by BERT are given as prompt, ∗ means golden topics are given as prompt*

|      Model       |   BLEU-2   | ROUGE-L  |   METEOR   |  DIST-2  | Topic ACC. |
| :--------------: | :--------: | :------: | :--------: | :------: | :--------: |
|   Transformer-   |   7.28%    |   0.21   |   0.1570   |   0.29   |     -      |
|      BART-       |   19.29%   |   0.35   |   0.2866   |   0.09   |     -      |
|       CPT-       |   19.79%   |   0.36   |   0.2969   |   0.07   |     -      |
|   Transformer    |   13.43%   |   0.33   |   0.2620   |   0.04   |   36.82%   |
|       BART       |   28.62%   |   0.48   |   0.4053   |   0.07   |   59.56%   |
|       CPT        |   29.40%   |   0.48   |   0.4142   |   0.06   |   59.77%   |
| Transformer-BERT |   23.95%   |   0.40   |   0.3758   |   0.22   |   61.32%   |
|    BART-BERT     |   33.73%   |   0.50   |   0.4598   |   0.07   |   61.32%   |
|     CPT-BERT     |   34.64%   |   0.51   |   0.4671   |   0.06   |   61.32%   |
|   Transformer*   |   25.37%   |   0.41   |   0.3905   |   0.04   |     -      |
|      BART*       |   37.02%   |   0.54   |   0.4920   |   0.07   |     -      |
|     **CPT***     | **37.45%** | **0.54** | **0.4943** | **0.06** |     -      |

### Dialogue Summary

|  Model  |   BLEU-2   | ROUGE-L  |  METEOR  |  DIST-2  | Symptom F1 |
| :-----: | :--------: | :------: | :------: | :------: | :--------: |
|  BART   |   16.44%   |   0.26   |   0.25   |   0.19   |    0.67    |
| **CPT** | **16.45%** | **0.26** | **0.24** | **0.21** |  **0.68**  |

### Severity Classification

#### depression severity

|  Task   |  Input  |  Model  |  Precision   |    Recall    |      F1      |
| :-----: | :-----: | :-----: | :----------: | :----------: | :----------: |
| 2-class | Dialog  |  BERT   |   0.81±.04   |   0.80±.03   |   0.80±.03   |
|         |         |  BART   |   0.80±.02   |   0.79±.03   |   0.79±.03   |
|         |         |   CPT   |   0.79±.02   |   0.78±.03   |   0.78±.03   |
|         | Summary |  BERT   |   0.90±.02   |   0.90±.02   |   0.90±.02   |
|         |         |  BART   |   0.89±.03   |   0.89±.03   |   0.89±.03   |
|         |         | **CPT** | **0.92±.01** | **0.92±.02** | **0.92±.01** |
| 4-class | Dialog  |  BERT   |   0.49±.05   |   0.45±.04   |   0.45±.04   |
|         |         |  BART   |   0.53±.04   |   0.53±.04   |   0.52±.04   |
|         |         |   CPT   |   0.49±.04   |   0.47±.04   |   0.46±.05   |
|         | Summary |  BERT   |   0.67±.04   |   0.66±.04   |   0.66±.04   |
|         |         |  BART   |   0.68±.03   |   0.67±.02   |   0.66±.02   |
|         |         | **CPT** | **0.73±.03** | **0.72±.03** | **0.72±.03** |

#### Suicide severity

|  Task   | Input  |  Model  |  Precision   |    Recall    |      F1      |
| :-----: | :----: | :-----: | :----------: | :----------: | :----------: |
| 2-class | Dialog |  BERT   |   0.81±.02   |   0.78±.02   |   0.79±.02   |
|         |        |  BART   |   0.77±.02   |   0.75±.02   |   0.75±.02   |
|         |        | **CPT** | **0.84±.02** | **0.82±.03** | **0.82±.03** |
| 4-class | Dialog |  BERT   |   0.72±.03   |   0.64±.04   |   0.66±.03   |
|         |        |  BART   |   0.70±.05   |   0.66±.04   |  0.65 ±.03   |
|         |        | **CPT** | **0.76±.02** | **0.68±.02** | **0.70±.02** |

## Requirements
The required python packages is listed in "requirements.txt". You can install them by
```
pip install -i requirements.txt
```
or 
```
conda install --file requirements.txt
```
## Raw Data Format
```
{
  "log":[ #dialog history
    {
      "text": "医生你好",
      "action": null,
      "speaker": "patient",
      "topic": []
    },
    {
        "text": "你好",
        "action": "其它", #topic
        "speaker": "doctor", 
        "topic": []
    },
  ],
  "portrait": #corresponding portrait of the dialog
  {
    "drisk": 3, #depression severity [0-5]
    "srisk": 2, #suicide severity [0-4]
    "age": "18",
    "gender": "男",
    "martial_status": "未婚",
    "occupation": "无职业",
    "symptoms": "决断困难；睡眠",
    "reason": ""
  },
  "record": #medical record of the dialog
  {
    "drisk": 2,
    "srisk": 2,
    "summary": "来访者近两周烦躁，有不合理的自罪想法，有自杀观念和行为。"
  }
}
```
## Data preprocess
Preprocess the raw data for dialog and summary
```
bash ./scripts/preprocess_data.sh
```

## Reference
If you use any source codes or datasets included in this repository in your work, please cite the corresponding paper. The bibtex are listed below (will be updated after formal public):
```
@article{yao2022d4,
  title={D4: a Chinese Dialogue Dataset for Depression-Diagnosis-Oriented Chat},
  author={Yao, Binwei and Shi, Chao and Zou, Likai and Dai, Lingfeng and Wu, Mengyue and Chen, Lu and Wang, Zhen and Yu, Kai},
  journal={arXiv preprint arXiv:2205.11764},
  year={2022}
}
```