set -x


GPUS=${1}
GPUS_PER_NODE=${2}
JOB_NAME=${3}
QUOTATYPE=${4}
PARTITION=${5}

# GPUS=${GPUS:-8}
# GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-10}

if [ $GPUS -lt 8 ]; then
    NODE=1
else
    NODE=$[GPUS/GPUS_PER_NODE]
fi

SCRIPT=${6}
CONFIG=${7}

CFGNAME=`basename ${CONFIG} .yaml`
SCRIPTNAME=`basename ${SCRIPT} .py`
DIR=./OUTPUT/${CFGNAME}
mkdir -p ${DIR}
SUFFIX=`date '+%Y%m%d%H%M'`

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${PY_ARGS:-""}

export DISABLE_ADDMM_CUDA_LT=1
export MASTER_PORT=22115
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1
srun -p ${PARTITION} \
    --quotatype=${QUOTATYPE} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u ${SCRIPT} --config_file=${CONFIG} --output_dir=${DIR} --run_name ${CFGNAME} \
    ${@:8} ${PY_ARGS} 2>&1 | tee -a ${DIR}/${SCRIPTNAME}_${SUFFIX}.log
#done