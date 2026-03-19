#!/bin/bash
# cd /home/yu/XRT/MultipleTiling/cond/2rxs_remove

# cat > run_2rxs_one.sh << 'EOF'

# cd "$(dirname "$0")"

# ICDIR="$1"
# INDEX="$2"

# echo "Host: $(hostname)"
# echo "ICDIR: $ICDIR  INDEX: $INDEX"
# date

# python3 2rxs_remove_one.py "$ICDIR" "$INDEX"

# echo "Done."
# date
# EOF

# chmod +x run_2rxs_one.sh

source /home/yu/XRT/.venv/bin/activate
mkdir output
python3 TS_beta_rho_gamma5.py "$1" "$2" "$3"

# source /home/yu/XRT/.venv/bin/activate
# mkdir output
# python3 2rxs_remove_one.py "$ICDIR" "$INDEX"