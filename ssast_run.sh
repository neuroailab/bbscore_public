## LeBel2023AudioTRUTS01
# 3rd layer
python run.py --model wav2vec2_base --layer _orig_mod.encoder.layers.2  --benchmark LeBel2023AudioTRUTS01 --metric temporal_rsa --a1-only > ../results/temporal_rsa/wav2vec2_01_layer2.txt

python run.py --model ssast_frame --layer ast.v.blocks.2 --benchmark LeBel2023AudioTRUTS01 --metric temporal_rsa --a1-only > ../results/temporal_rsa/ssast_01_layer2.txt

# last layer
python run.py --model wav2vec2_base --layer _orig_mod.encoder.layers.11.final_layer_norm  --benchmark LeBel2023AudioTRUTS01 --metric temporal_rsa --a1-only > ../results/temporal_rsa/wav2vec2_01_layer11_final_norm.txt

python run.py --model ssast_frame --layer ast.v.norm --benchmark LeBel2023AudioTRUTS01 --metric temporal_rsa --a1-only > ../results/temporal_rsa/ssast_01_v_norm.txt

## LeBel2023AudioTRUTS02 
# 3rd layer
python run.py --model wav2vec2_base --layer _orig_mod.encoder.layers.2  --benchmark LeBel2023AudioTRUTS02 --metric temporal_rsa --a1-only > ../results/temporal_rsa/wav2vec2_02_layer2.txt

python run.py --model ssast_frame --layer ast.v.blocks.2 --benchmark LeBel2023AudioTRUTS02 --metric temporal_rsa --a1-only > ../results/temporal_rsa/ssast_02_layer2.txt

# last layer
python run.py --model wav2vec2_base --layer _orig_mod.encoder.layers.11.final_layer_norm  --benchmark LeBel2023AudioTRUTS02 --metric temporal_rsa --a1-only > ../results/temporal_rsa/wav2vec2_02_layer11_final_norm.txt

python run.py --model ssast_frame --layer ast.v.norm --benchmark LeBel2023AudioTRUTS02 --metric temporal_rsa --a1-only > ../results/temporal_rsa/ssast_02_v_norm.txt

## LeBel2023AudioTRUTS03
# 3rd layer
python run.py --model wav2vec2_base --layer _orig_mod.encoder.layers.2  --benchmark LeBel2023AudioTRUTS03 --metric temporal_rsa --a1-only > ../results/temporal_rsa/wav2vec2_03_layer2.txt

python run.py --model ssast_frame --layer ast.v.blocks.2 --benchmark LeBel2023AudioTRUTS03 --metric temporal_rsa --a1-only > ../results/temporal_rsa/ssast_03_layer2.txt

# last layer
python run.py --model wav2vec2_base --layer _orig_mod.encoder.layers.11.final_layer_norm  --benchmark LeBel2023AudioTRUTS03 --metric temporal_rsa --a1-only > ../results/temporal_rsa/wav2vec2_03_layer11_final_norm.txt

python run.py --model ssast_frame --layer ast.v.norm --benchmark LeBel2023AudioTRUTS03 --metric temporal_rsa --a1-only > ../results/temporal_rsa/ssast_03_v_norm.txt