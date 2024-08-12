
## Environment Installation

```bash
pip install -r requirements.txt
```

## Dataset Preparation

### Training dataset

1. To download DF2K and DIV2K validation

   Make the data tree like this

   ```
   data/sr
   ├── DF2K
   │   └── DF2K_train_HR
   │       ├── 0002.png
   │       ├── 0003.png
   │       ├── 0001.png
   |       ├── ...
   └── DIV2K
       └── DIV2K_valid_HR
           ├── 0002.png
           ├── 0003.png
           ├── 0001.png
           ├── ...
   ```


2. Generate sam mask
   
    1. download [segment-anything](https://github.com/facebookresearch/segment-anything) code, and download the *
       *`vit_h`** checkpoint.

       ```shell
       git clone https://github.com/facebookresearch/segment-anything.git
       ```

    2. generate mask data in RLE format by sam

          ```shell
          python scripts/amg.py \
          --checkpoint weights/sam_vit_h_4b8939.pth \
          --model-type vit_h \
          --input data/sr/DF2K/DF2K_train_HR \
          --output data/sam_out/DF2K/DF2K_train_HR \
          --convert-to-rle
          ```

    3. use SPE to embedded the RLE format mask

          ```shell
          python scripts/merge_mask_to_one.py \
          --input data/sam_out/DF2K/DF2K_train_HR \
          --output data/sam_embed/DF2K/DF2K_train_HR
          ```

4. build bin dataset

   ```shell
   python data_gen/df2k.py --config configs/data/df2k4x_sam.yaml
   ```

### Benchmark dataset

1. download the dataset. e.g Set5, Set14, Urban100, Manga109, BSDS100

2. change the `data_name` and `data_path` in `data_gen/benchmark.py`, and run:

   ```
   python data_gen/benchmark.py --config configs/data/df2k4x_sam.yaml
   ```

## Training

1. download rrdb pretrain model from https://github.com/LeiaLi/SRDiff/releases/tag/v1.0.0, and move
   the weight to `./weights/rrdb_div2k.ckpt`

2. train diffusion model

   ```shell
   python tasks/trainer.py \
   --config configs/sam/sam_diffsr_df2k4x.yaml \
   --exp_name sam_diffsr_df2k4x \
   --reset \
   --hparams="rrdb_ckpt=weights/rrdb_div2k.ckpt" \
   --work_dir exp/
   ```

## Evaluation

- evaluate specified checkpoint (like 400000 steps)

  ```shell
  python tasks/trainer.py 
  --benchmark \
  --hparams="test_save_png=True" \
  --exp_name sam_diffsr_df2k4x \
  --val_steps 400000 \
  --benchmark_name_list test_Set5 test_Set14 test_Urban100 test_Manga109 test_BSDS100
  ```

  If you want to replicate our results, you should download the checkpoint and move it
  to `SAM-DiffSR/checkpoints/sam_diffsr_df2k4x` directory.

- evaluate all checkpoint

  ```shell
  python tasks/trainer.py \
  --benchmark_loop \
  --exp_name sam_diffsr_df2k4x \
  --benchmark_name_list test_Set5 test_Set14 \
  ```

## Inference

```shell
python tasks/infer.py \
--config configs/sam/sam_diffsr_df2k4x.yaml \
--img_dir your/lr/img/path \
--save_dir your/sr/img/save/path \
--ckpt_path model_ckpt_steps_400000.ckpt
```
