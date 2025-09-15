# EMO-COT Inference Code

![Description of image](assets/1.pdf)


## Training and Inference

```bash
To fine-tune easop with grad-tts
accelerate launch --config_file accelerate_cfg/1a4o_fp16.yaml train_scripts/train_easpo.py --config configs/easpo_-v1-5_4k-prompts_num-sam-4_10ep_bs10.py
```

```bash
To inference easop with grad-tts
python inference_scripts/inference.py
```
