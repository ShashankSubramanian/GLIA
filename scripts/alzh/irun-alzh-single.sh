ALZH=/scratch1/04678/scheufks/alzh/
RES=${ALZH}/real_test/
mkdir -p ${RES}
python3  run-real.py       -cluster frontera \
                      -x        ${RES}   
