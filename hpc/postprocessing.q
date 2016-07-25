#PBS -l nodes=1:ppn=8
#PBS -l walltime=0:30:00
#PBS -l mem=6GB
#PBS -M tom.huang@nyu.edu
#PBS -j oe

module purge
module load anaconda/2.3.0
module load python3/intel/3.5.1
# SRCDIR=$HOME/nyu-twipsy

RUNDIR=$SCRATCH/nyu-twipsy/run-8754569
# mkdir -p $RUNDIR/

# cp -r $SRCDIR $RUNDIR
cd $RUNDIR

# rm -r $RUNDIR/nyu-twipsy/hpc/in/
# mkdir -p $RUNDIR/nyu-twipsy/hpc/in/
# cp -r $HOME/sept-blocked/. $RUNDIR/nyu-twipsy/hpc/in

PROJECT=$RUNDIR/nyu-twipsy

PYTHONPATH=$PROJECT python3 $PROJECT/hpc/postprocessing.py $PROJECT/hpc/out $PROJECT/hpc/summary 8

