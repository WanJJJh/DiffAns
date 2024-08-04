for param1 in 50; do
        for param2 in 1; do
                python3 train.py \
                        --dataset tgif-qa \
                        --question_type frameqa \
                        --model_id 0 \
                        --layer_num 3 \
                        --T $param1 \
                        --scale 0.5\
                        --lm_name deberta-base \
                        --lm_frozen 0 \
                        --num_frames 16 \
                        --gpu_id 0 \
                        --max_epochs 10 \
                        --batch_size 128 \
                        --dropout 0.3 \
                        --use_train \
                        --use_test
        done;
done