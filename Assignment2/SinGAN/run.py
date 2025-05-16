import subprocess
import shlex
import os

image_train = "cows.png"
image_eval = "cowEval.png"
results_dir = "./results"

# Generator and Discriminator models (set dynamically)
gen_model = ["g_multipixelattention", "g_multivanilla"]
dis_model = ["d_snpixelattention", "d_vanilla"]

currGenModel = gen_model[1]
currDisModel = dis_model[1]

# model_to_load = "results/cows$2025-04-25_09-55-28/g_multivanilla.pth"
# safe_model_path = shlex.quote(model_to_load)
# amps_to_load = "results/cows$2025-04-25_09-55-28/amps.pth"
# safe_amps_path = shlex.quote(amps_to_load)

# Iterate over each signature and execute the command

command = f"python main.py --root images/{image_train} --dir-name {image_train.split('.')[0]} --gen-model {currGenModel} --dis-model {currDisModel} \
    --print-every 400 --use-tb \
    --noise-weight 0.01 \
    --seed 42 --num-steps 4000 > outLog_{image_train.split('.')[0]}.log 2>&1"
    
# command = f"""
#     python main.py --root images/{image_train} --evaluation \
#     --dir-name Eval{image_train.split('.')[0]} \
#     --model-to-load {safe_model_path} \
#     --noise-weight 0.01 \
#     --gen-model {currGenModel} --dis-model {currDisModel} \
#     --amps-to-load {safe_amps_path} \
#     --num-steps 1 --batch-size 8 > evalLog_{image_eval.split('.')[0]}.log 2>&1
#     """

print(f"Executing: {command}")
subprocess.run(command, shell=True, check=True)
