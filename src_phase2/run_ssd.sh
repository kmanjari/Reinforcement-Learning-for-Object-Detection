# Note: all paths referenced here are relative to the Docker container.
#
# Add the Nvidia drivers to the path
export PATH="/usr/local/nvidia/bin:$PATH"
export HOME="/storage/home/sidnayak"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib:$LD_LIBRARY_PATH"
# Tools config for CUDA, Anaconda installed in the common /tools directory
source /tools/config.sh
# Activate your environment
source activate py35
# Change to the directory in which your code is present
cd /storage/home/sidnayak/Reinforcement-Learning-for-Object-Detection/src_phase2
# Run the code. The -u option is used here to use unbuffered writes
# so that output is piped to the file as and when it is produced.

#python -u ssdAgentDetect.py --load_model=1 --epoch=100 --lr=1e-6 &> outputs/ssd_training_50
#python -u ssd_random.py --epoch=20 &> outputs/out_random_newReward
python -u ssd_agentDetect_newReward.py --load_model=0 --epoch=1 &> outputs/out_ssd_newReward
#python -u ssd_baseline.py --epoch=1 &> outputs/out_baseline_newReward
