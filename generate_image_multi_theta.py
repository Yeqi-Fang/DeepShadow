from pathlib import Path

root_dir = Path('tele_datasets\Changing')

output_dir = Path('tele_datasets\mixed')

for i in root_dir.iterdir():
    print(i)