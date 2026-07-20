
# MoSTE: Adaptively Routed Mixture of Spatial and Temporal Experts for Multi-Site Water Quality Forecasting

This repository provides the official PyTorch implementation of MoSTE in the paper **"Adaptively Routed Mixture of Spatial and Temporal Experts for Multi-Site Water Quality Forecasting"**.

## Requirements

- Python >= 3.9
- PyTorch >= 1.13.1
- NumPy >= 1.25.2
- pandas >= 2.2.2
- scikit-learn >= 1.5.2

Dependencies can be installed using the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Model Training

MoSTE uses multi-stage training: Expert Capability Learning, Routing Preference Learning, and Joint Fine-Tuning. The detailed procedure follows Section IV-E of the paper and is summarized in the table below.

<img width="525" height="237" alt="b7775a176b1686fadebf165473af5182" src="https://github.com/user-attachments/assets/268e562f-75dd-4930-b08e-7b53cf994699" />

#### Stage 1: Expert Capability Learning

Stage 1 consists of three independent training runs, one for each expert. Each run trains one expert as a standalone forecasting model rather than using the complete MoSTE framework. For example, when training the Temporal Expert, the Prior Graph Expert, Latent Graph Expert, and router are disabled in the model code, with the Bash script adjusted accordingly to include only the command-line arguments required for the Temporal Expert. The same setup is applied separately to the Prior Graph Expert and Latent Graph Expert. Save the best checkpoint from each run in `checkpoints/`, resulting in three expert checkpoints.

#### Stage 2: Routing Preference Learning

Stage 2 uses the complete MoSTE model code, with all three expert branches and the dual-branch router enabled. Load the three expert checkpoints obtained in Stage 1 into their corresponding expert branches. Freeze all expert parameters so that the experts participate only in the forward pass. Train only the router and save the best Stage 2 checkpoint in `checkpoints/`.

#### Stage 3: Joint Fine-Tuning

Stage 3 keeps all three expert branches and the dual-branch router enabled in the model code. Load the three expert checkpoints from Stage 1 and the router parameters from the Stage 2 checkpoint into the complete MoSTE framework. Unfreeze all expert and router parameters, and then use the complete-model command template shown below to jointly train the entire model. The checkpoint produced by this stage is the final MoSTE model.

```bash
python -u run.py \
  --task_name TASK_NAME \
  --is_training 1 \
  --root_path ROOT_PATH \
  --data_path DATA_PATH \
  --prior_graph_path PRIOR_GRAPH_PATH \
  --model_id MODEL_ID \
  --model MODEL \
  --data DATA \
  --features FEATURES \
  --target TARGET \
  --freq FREQ \
  --seq_len SEQ_LEN \
  --label_len LABEL_LEN \
  --pred_len PRED_LEN \
  --enc_in ENC_IN \
  --c_out C_OUT \
  --d_model D_MODEL \
  --e_layers E_LAYERS \
  --d_ff D_FF \
  --down_sampling_layers DOWN_SAMPLING_LAYERS \
  --down_sampling_method DOWN_SAMPLING_METHOD \
  --down_sampling_window DOWN_SAMPLING_WINDOW \
  --steps_per_day STEPS_PER_DAY \
  --structure_weight STRUCTURE_WEIGHT \
  --batch_size BATCH_SIZE \
  --train_epochs TRAIN_EPOCHS \
  --patience PATIENCE \
  --learning_rate LEARNING_RATE \
  --des DESCRIPTION \
  --itr ITERATIONS
```

### Model Testing

To evaluate the final checkpoint, set `--is_training` to `0` and retain the data and model configuration used in Stage 3. Training-only parameters are not required.

```bash
python -u run.py \
  --task_name TASK_NAME \
  --is_training 0 \
  --root_path ROOT_PATH \
  --data_path DATA_PATH \
  --prior_graph_path PRIOR_GRAPH_PATH \
  --model_id MODEL_ID \
  --model MODEL \
  --data DATA \
  --features FEATURES \
  --target TARGET \
  --freq FREQ \
  --seq_len SEQ_LEN \
  --label_len LABEL_LEN \
  --pred_len PRED_LEN \
  --enc_in ENC_IN \
  --c_out C_OUT \
  --d_model D_MODEL \
  --e_layers E_LAYERS \
  --d_ff D_FF \
  --down_sampling_layers DOWN_SAMPLING_LAYERS \
  --down_sampling_method DOWN_SAMPLING_METHOD \
  --down_sampling_window DOWN_SAMPLING_WINDOW \
  --steps_per_day STEPS_PER_DAY \
  --structure_weight STRUCTURE_WEIGHT \
  --des DESCRIPTION
```
