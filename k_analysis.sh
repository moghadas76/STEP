set -ex



src=$1
k=$2
#
#for k in {0..0}
#do
#  declare script_path='/home/seyed/miniconda3/bin/python test/test_inference.py --cfg='\''step/STEP_METR-LA.py'\'' --ckpt='\''/home/seyed/PycharmProjects/step/STEP/checkpoints/STEP_100/f8cbc6b5dc005fca0774a226a394b6d9/STEP_best_val_MAE.pt --src="$src" -k="$k"
#  echo "Running ...: ${script_path}"
#  result=`$script_path`
#  /home/seyed/miniconda3/bin/python test/test_inference.py --cfg='step/STEP_METR-LA.py' --ckpt='/home/seyed/PycharmProjects/step/STEP/checkpoints/STEP_100/f8cbc6b5dc005fca0774a226a394b6d9/STEP_best_val_MAE.pt' --src="$src" -k="$k"
#done

#/home/seyed/miniconda3/bin/python test/test_inference.py --cfg='step/STEP_METR-LA.py' --ckpt='/home/seyed/PycharmProjects/step/STEP/checkpoints/STEP_100/f8cbc6b5dc005fca0774a226a394b6d9/STEP_best_val_MAE.pt' --src="0" -k="1" 2> out_k_1_src_0.log
declare var

va=$(/home/seyed/miniconda3/bin/python test/test_inference.py --cfg='step/STEP_METR-LA.py' --ckpt='/home/seyed/PycharmProjects/step/STEP/checkpoints/STEP_100/f8cbc6b5dc005fca0774a226a394b6d9/STEP_best_val_MAE.pt' --src=$src -k=$k 2>&1)
var=$(echo -e $va | /home/seyed/miniconda3/bin/python <(cat <<HERE
import sys
lst = None
for i, line in enumerate(sys.stdin):
  lst = line.rstrip()
result = float(lst.split("test_MAE:")[2].split(",")[0])
print(result)
HERE
))
echo -e "$src, $k, $var\n" | tee -a /home/seyed/PycharmProjects/step/STEP/test_errors.log
