echo "start training sp + lg!"
CUDA_VISIBLE_DEVICES=0 python -m gluefactory.train demo_xfeat_retina \
--conf ../training/configs/xfeat_retina_homography.yaml  \