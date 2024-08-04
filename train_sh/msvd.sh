for param1 in 50; do
        for param2 in 1; do
                python3 train.py \
                        --dataset msvd-qa \
                        --question_type none \
                        --model_id 0 \
                        --layer_num 4 \
                        --T $param1 \
                        --scale 0.5 \
                        --lm_name roberta-base \
                        --lm_frozen 0 \
                        --num_frames 16 \
                        --gpu_id 0 \
                        --max_epochs 10 \
                        --batch_size 1 \
                        --dropout 0.3 \
                        --use_test
        done ;
done