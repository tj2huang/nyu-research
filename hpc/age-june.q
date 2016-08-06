#PBS -l nodes=1:ppn=16
#PBS -l walltime=8:00:00
#PBS -l mem=32GB
#PBS -M tom.huang@nyu.edu
#PBS -j oe

module purge

module load python3/intel/3.5.1
module load nltk/3.0.2

SRCDIR=$HOME/nyu-twipsy

RUNDIR=$SCRATCH/nyu-twipsy/run-age-june-${PBS_JOBID/.*}
mkdir -p $RUNDIR/

cp -r $SRCDIR $RUNDIR/
cd $RUNDIR

PROJECT=$RUNDIR/nyu-twipsy
INFOLDER=$SCRATCH/nyu-twipsy/data/june

PYTHONPATH=$PYTHONPATH:$PROJECT python3 $PROJECT/hpc/split_age.py $INFOLDER $PROJECT/hpc/out 16

