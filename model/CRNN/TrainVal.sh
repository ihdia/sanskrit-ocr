#! /bin/bash
# int-or-string.sh
#data/tfReal/
a=
let b=a-1
echo $b
k=203
c=1
#load model from:
#CUDA_VISIBLE_DEVICES=0 python3 tools/train_shadownet.py --dataset_dir data/tfReal/ --train_epochs $k --weights_path model/shadownet/shadownet_-8956
#else fresh model
#python3 tools/train_shadownet.py --dataset_dir data/tfReal/ --train_epochs $a
#python3 tools/test_shadownet.py --dataset_dir data/tfReal/ --weights_path $(printf "model/shadownet/shadownet_-%d" "$b")
for i in {1..400}; do
echo model/CRNN/model/shadownet/shadownet_-$b
	CUDA_VISIBLE_DEVICES=0 python3 tools/test_shadownet.py --dataset_dir data/tfReal/ --weights_path $(printf "model/shadownet/shadownet_-%d" "$b")
	cp $(printf "model/CRNN/model/shadownet/shadownet_-%d.data-00000-of-00001" "$b") $(printf "model/CRNN/model/shadownet_-%d.data-00000-of-00001" "$c");
	cp $(printf "model/CRNN/model/shadownet/shadownet_-%d.meta" "$b") $(printf "model/CRNN/model/shadownet_-%d.meta" "$c");
	cp $(printf "model/CRNN/model/shadownet/shadownet_-%d.index" "$b") $(printf "model/CRNN/model/shadownet_-%d.index" "$c");
	echo $(printf "model/shadownet/shadownet_-%d.index" "$c")	
	let c=c+1
	CUDA_VISIBLE_DEVICES=0 python3 tools/train_shadownet.py --dataset_dir data/tfReal/ 
	let b=a-1
	CUDA_VISIBLE_DEVICES=0 python3 tools/test_shadownet.py --dataset_dir data/tfReal/ --weights_path $(printf "model/shadownet/shadownet_-%d" "$b") #>> chec_acc.txt
done
#./TrainVal.sh > 005Mix/logTrainVal 2> 005Mix/logTrainVal2
#grep -r "Test accuracy is" logTrainVal
