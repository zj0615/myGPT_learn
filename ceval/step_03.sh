model_path=/workspace/myGPT/ckpt/sft/epoch6_batch_39999
vocal_path=/workspace/myGPT/model/sft/vocab.txt
output_path=/workspace/myGPT/ceval/output/pretain/

python eval.py \
    --model_path ${model_path} \
    --vocab_path ${vocal_path} \
    --cot False \
    --few_shot False \
    --with_prompt True \
    --constrained_decoding  False \
    --n_times 1 \
    --ntrain 5 \
    --temperature 0.9 \
    --do_save_csv False \
    --do_test False \
    --output_dir ${output_path} \
    --device 0