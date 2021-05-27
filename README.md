# Paper Evaluator

We design several models to evaluate papers from different perspectives, including **appearance**, **text coherence**, **structure** and element **diversity**.

## Models

- Deep Paper Gestalt

  <center><img src="figures\paper_gestalt.png" width="500"/></center>

- Gradient Boosting Decision Tree

  <center><img src="figures\lightgbm.png" width="500"/></center>

- Paper-sequenced based LSTM

  <center><img src="figures\paper_seq.png" width="500"/></center>

  <center><img src="figures\seq_lstm.png" width="500"/></center>

- Attention-based RCNN

  <center><img src="figures\att_rcnn.png" width="500"/></center>

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

