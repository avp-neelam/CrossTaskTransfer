# CrossTaskTransfer

## Running Experiments

Single dataset
```bash
python run.py \
  --root /path/to/data \
  --family Planetoid \
  --name Cora \
  --seeds 1 2 3
```

Full experiment
```bash
python main.py --run_all --setting transductive --seeds 1 2 3 4 5 6 7 8 9 10 --enable_prompt_transfer --enable_joint_baseline
```
