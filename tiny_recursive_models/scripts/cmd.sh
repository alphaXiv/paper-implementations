#Pretrain MAZE-HARD

torchrun --nproc-per-node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py arch=trm data_paths="[data/maze-30x30-hard-1k]" evaluators="[]" epochs=50000 eval_interval=5000 global_batch_size=1536 lr=2e-4 lr_warmup_steps=4000 puzzle_emb_lr=1e-4 checkpoint_every_eval=True weight_decay=1.0 puzzle_emb_weight_decay=1.0 arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 +run_name=${run_name} ema=True

#Sudoko-extreme

#mlp
run_name="pretrain_mlp_t_sudoku"

torchrun --nproc-per-node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=2e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
lr_warmup_steps=4000 \
global_batch_size=1536 \ 
+run_name=${run_name} ema=True

#Attn

torchrun --nproc-per-node=8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py arch=trm data_paths="[data/sudoku-extreme-1k-aug-1000]" evaluators="[]" epochs=50000 eval_interval=5000 lr=2e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 lr_warmup_steps=4000 global_batch_size=1536 +run_name=${run_name} ema=True


---
#evals MAZE-HARD

#Attn

torchrun --nproc_per_node=8 run_eval.py \
  --checkpoint ./step_32550 \
  --dataset data/maze-30x30-hard-1k \
  --outdir checkpoints/maze_eval_run \
  --eval-save-outputs inputs labels puzzle_identifiers preds \
  --global-batch-size 1536 \
  --apply-ema \
  --repeats 3 \
  --seed-start 0

#evals Sudoku

#MLP
torchrun --nproc_per_node=8 run_eval.py \
  --checkpoint ./step_32550_sudoku_epoch50k \
  --dataset data/maze-30x30-hard-1k \
  --outdir checkpoints/maze_eval_run \
  --eval-save-outputs inputs labels puzzle_identifiers preds \
  --global-batch-size 1536 \
  --apply-ema \
  --repeats 3 \
  --seed-start 0



### ARC-AGI - 1

# attention
run_name="pretrain_att_arc1concept_4"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py arch=trm data_paths="[data/arc1concept-aug-1000]" arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 +run_name=${run_name} ema=True lr=2e-4 weight_decay=0.1 global_batch_size=1536 lr_warmup_steps=4000 epochs=100000 puzzle_emb_lr=1e-2 eval_interval=5000

MLPrun_name="pretrain_att_arc1concept_4"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py arch=trm data_paths="[data/arc1concept-aug-1000]" arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 +run_name=${run_name} ema=True lr=2e-4 weight_decay=0.1 global_batch_size=1536 lr_warmup_steps=4000 epochs=100000 puzzle_emb_lr=1e-2 eval_interval=5000

# MLP

run_name="pretrain_att_arc1concept_h3l6"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py arch=trm data_paths="[data/arc1concept-aug-1000]" arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 +run_name=${run_name} ema=True lr=2e-4 weight_decay=0.1 global_batch_size=1536 lr_warmup_steps=4000 epochs=100000 puzzle_emb_lr=1e-2 eval_interval=5000 arch.mlp_t=true

  