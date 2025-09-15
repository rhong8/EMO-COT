# EMO-COT Inference Code

![Model Diagram](https://github.com/jiachengQAQ/EMO-COT/blob/main/assets/1.png?raw=true)


## Training and Inference

```bash
To build emotion graph on iemocap
python iemocap/build_graph.py
```

```bash
To inference emotion with LALMs
python eval_audio/evaluate_emotion_local.py --dataset iemocap --checkpoint Qwen/Qwen2-Audio-7B
```
