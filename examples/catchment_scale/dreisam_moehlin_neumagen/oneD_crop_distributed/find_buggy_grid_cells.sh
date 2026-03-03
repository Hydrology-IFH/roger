#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=find_buggy_grid_cells
#SBATCH --output=find_buggy_grid_cells.out
#SBATCH --error=find_buggy_grid_cells_err.out
#SBATCH --export=ALL

module load devel/miniforge
conda activate roger
cd /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed

python find_buggy_grid_cells.py