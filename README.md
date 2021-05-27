# Paper Evaluator

We design several models to evaluate papers from different perspectives, including **appearance**, **text coherence**, **structure** and element **diversity**.

## Models

- Deep Paper Gestalt

  ![image-20210527215006671](\figures\paper_gestalt.png)

- Gradient Boosting Decision Tree

  ![image-20210527215026542](\figures\lightgbm.png)

- Paper-sequenced based LSTM

  ![image-20210527215042140](\figures\paper_seq.png)

  ![image-20210527215058343](\figures\seq_lstm.png)

- Attention-based RCNN

  ![image-20210527215129878](\figures\att_rcnn.png)

## Dataset

CVPR 2015 - 2020 (you can use crawler.py to get these papers from CVFs on your own.)

## Performance

| **Methods**                     | **Accuracy** | **F1-score** |
| ------------------------------- | ------------ | ------------ |
| Paper-image+ResNet18            | 84%          | 76%          |
| Lightgbm                        | 87%          | 81%          |
| Paper-seq+CNN (Ours)            | 86.84%       | 77.96%       |
| Paper-seq+CNN LSTM (Ours)       | 89.41%       | 83.18%       |
| **Paper-seq+LSTM**  **(Ours)**  | **90.30%**   | **84.59%**   |
| **Attention-based RCNN (Ours)** | **88%**      | **81%**      |

