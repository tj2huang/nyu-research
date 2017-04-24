#PBS -l nodes=1:ppn=16
#PBS -l walltime=1:30:00
#PBS -l mem=12GB
#PBS -M tom.huang@nyu.edu
#PBS -j oe

PROCESSES=16

module purge
module load python3/intel/3.5.1
SRCDIR=$HOME/nyu-twipsy

RUNDIR=$SCRATCH/nyu-twipsy/run-june-under-${PBS_JOBID/.*}
mkdir -p $RUNDIR/

#cp -r $SRCDIR $RUNDIR/
#cd $RUNDIR

mkdir $RUNDIR/out
mkdir $RUNDIR/summary

#PROJECT=$RUNDIR/nyu-twipsy
PROJECT=$SRCDIR
INFOLDER=$SCRATCH/nyu-twipsy/data/june-age/under

PYTHONPATH=$PROJECT python3 $PROJECT/hpc/main.py $INFOLDER $RUNDIR/out $PROCESSES
PYTHONPATH=$PROJECT python3 $PROJECT/hpc/postprocessing.py $RUNDIR/out $RUNDIR/summary $PROCESSES

