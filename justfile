poetry-export-pip:
    poetry export --without-hashes --without-urls | awk '{ print $1 }' FS=';' > requirements.txt

cpu-run cmd cpus='8' mem='32G' time='6:00:00' *ARGS='':
    #!/bin/bash
    set -e

    tmp_dir=$HOME/tmp/ollm-$(date +%s)
    mkdir -p $tmp_dir

    echo "Copying files to $tmp_dir"
    git ls-files --cached --others --exclude-standard | xargs -I {} cp --parents {} $tmp_dir
    cp .env $tmp_dir
    ln -s /rds/user/cyal4/hpc-work/ollm/out $tmp_dir/out
    cd $tmp_dir

    out=$(echo {{cmd}} | sed 's/[^a-zA-Z0-9]/_/g')
    sbatch --cpus-per-task={{cpus}} --mem={{mem}} --time={{time}} --output=out/logs/$out-%j.out {{ARGS}} entrypoint/slurm/launch.sh {{cmd}}

    echo "Submitted command: {{cmd}} with {{cpus}} cpus, {{mem}} memory for {{time}} with args {{ARGS}}"

gpu-run cmd gpus='1' time='6:00:00' *ARGS='':
    #!/bin/bash
    set -e

    tmp_dir=$HOME/tmp/ollm-$(date +%s)
    mkdir -p $tmp_dir

    echo "Copying files to $tmp_dir"
    git ls-files --cached --others --exclude-standard | xargs -I {} cp -r --parents {} $tmp_dir
    cp .env $tmp_dir
    ln -s /rds/user/cyal4/hpc-work/ollm/out $tmp_dir/out
    cd $tmp_dir

    out=$(echo {{cmd}} | sed 's/[^a-zA-Z0-9]/_/g')
    sbatch --time={{time}} --gres=gpu:{{gpus}} --output=out/logs/$out-%j.out {{ARGS}} entrypoint/slurm/launch_gpu.sh {{cmd}}

    echo "Submitted command: {{cmd}} with {{gpus}} gpus for {{time}} with args {{ARGS}}"

intr-cpu cpus='4' mem='10G' time='1:00:00':
    just cpu-run 'sleep infinity' {{cpus}} {{mem}} {{time}} --qos=INTR

intr-gpu gpus='1' time='1:00:00':
    just gpu-run 'sleep infinity' {{gpus}} {{time}} --qos=INTR