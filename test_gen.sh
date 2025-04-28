# First approach: initialize 4 VLLM processes and split the prompt set to the 4 agents
# The generated samples will be stored at output_dir + local_index + ".jsonl
source ~/.bashrc

# Initialize Conda environment
eval "$(conda shell.bash hook)"

my_world_size=1 # how many gpu you use
infer_model=kasbar/LLaMA3.2-1B-INPO_iter1
prompt_dir=RLHFlow/ultrafeedback_iter2

mkdir data
output_dir=./data/gen_data

conda activate vllm2
CUDA_VISIBLE_DEVICES=0 python ./generation/gen_hf2.py --model_name_or_path ${infer_model} --dataset_name_or_path ${prompt_dir} --output_dir ${output_dir} --K 4 --temperature 1.0 --local_index 0 --my_world_size ${my_world_size}  &

# then, we merge the 8 datasets into one dataset.
wait
python ./generation/merge_data.py --base_path ${output_dir} --output_dir ./data/gen_data_iter2.json --num_datasets ${my_world_size}

