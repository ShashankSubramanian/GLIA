ALZH=/scratch1/04678/scheufks/alzh/
RES=${ALZH}/syn_test/test05
mkdir -p ${RES}
python3  run.py       -cluster frontera \
                      -x        ${RES}   
