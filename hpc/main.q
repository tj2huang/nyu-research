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

PROJECT=$RUNDIR/nyu-twipsy
INFOLDER=$SCRATCH/nyu-twipsy/data/june/

PYTHONPATH=$PROJECT python3 $PROJECT/hpc/main.py $PROJECT/hpc/classifiers/clf_alc_UPDATED.p $PROJECT/hpc/classifiers/clf_fpa_UPDATED.p $PROJECT/hpc/classifiers/clf_fpl_double_labeled $INFOLDER $PROJECT/hpc/out 16
PYTHONPATH=$PROJECT python3 $PROJECT/hpc/postprocessing.py $PROJECT/hpc/out $PROJECT/hpc/summary 16

