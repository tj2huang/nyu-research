#PBS -l nodes=1:ppn=16
#PBS -l walltime=8:00:00
#PBS -l mem=12GB
#PBS -M tom.huang@nyu.edu
#PBS -j oe

module purge
module load anaconda/2.3.0
module load python3/intel/3.5.1
SRCDIR=$HOME/nyu-twipsy

RUNDIR=$SCRATCH/nyu-twipsy/run-${PBS_JOBID/.*}
mkdir -p $RUNDIR/

cp -r $SRCDIR $RUNDIR/
cd $RUNDIR

# rm -r $RUNDIR/nyu-twipsy/hpc/in/
# mkdir -p $RUNDIR/nyu-twipsy/hpc/in/
# cp -r $HOME/sept-blocked/. $RUNDIR/nyu-twipsy/hpc/in

PROJECT=$RUNDIR/nyu-twipsy
INFOLDER=$SCRATCH/nyu-twipsy/data/june/march/

PYTHONPATH=$PROJECT python3 $PROJECT/hpc/main.py $PROJECT/hpc/classifiers/clf_alc_UPDATED.p $PROJECT/hpc/classifiers/clf_fpa_UPDATED.p $PROJECT/hpc/classifiers/clf_fpl_double_labeled $INFOLDER $PROJECT/hpc/out 16

