ALZH=/scratch1/04678/scheufks/alzh/
RES=${ALZH}/real_test/
mkdir -p ${RES}
python3  run-real2.py       -cluster frontera \
                      -x        ${RES}   
