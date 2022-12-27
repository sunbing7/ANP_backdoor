#cifar
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=resnet18 --poison_type=semantic --ana_layer 6 --plot=0 --batch_size=64 --num_sample=256 --poison_target=6 --in_model=./save/model_semtrain_green_last.th --output_dir=./save --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=resnet18 --poison_type=semantic --confidence=5 --ana_layer 6 --batch_size=64 --num_sample=256 --poison_target=6 --in_model=./save/model_semtrain_green_last.th --output_dir=./save --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
#python semantic_mitigation.py --option=gen_trigger --lr=0.1 --potential_source=1 --poison_target=6 --reg=0.9 --epoch=2000  --reanalyze=0 --arch=resnet18 --poison_type=semantic --batch_size=64 --num_sample=100 --in_model=./save/model_semtrain_green_last.th --output_dir=./save --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_dir=./data/CIFAR10 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=remove4 --lr=0.006 --reg=0.01 --epoch=6  --reanalyze=0 --arch=resnet18 --poison_type=semantic --confidence=3 --ana_layer 6 --batch_size=64 --potential_source=1 --poison_target=6 --in_model=./save/model_semtrain_green_last.th --output_dir=./save --t_attack=green --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=resnet18 --load_type=model --poison_type=semantic --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --poison_target=6 --in_model=./save/model_finetune4_green_last.th --output_dir=./save --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=resnet18 --load_type=model --potential_target=6 --poison_type=semantic --confidence=5 --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --poison_target=6 --in_model=./save/model_finetune4_green_last.th --output_dir=./save --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10

#cifar2
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=resnet18 --poison_type=semantic --ana_layer 6 --plot=0 --batch_size=64 --num_sample=256 --poison_target=9 --in_model=./save/model_semtrain_sbg_last.th --output_dir=./save --t_attack=sbg --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=resnet18 --poison_type=semantic --confidence=5 --ana_layer 6 --batch_size=64 --num_sample=256 --poison_target=9 --in_model=./save/model_semtrain_sbg_last.th --output_dir=./save --t_attack=sbg --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
#python semantic_mitigation.py --option=gen_trigger --lr=0.1 --potential_source=1  --poison_target=9 --reg=0.9 --epoch=2000  --reanalyze=0 --arch=resnet18 --poison_type=semantic --batch_size=64 --num_sample=100 --in_model=./save/model_semtrain_sbg_last.th --output_dir=./save --t_attack=sbg --data_set=./data/CIFAR10/cifar_dataset.h5 --data_dir=./data/CIFAR10 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=remove4 --lr=0.006 --reg=0.02 --epoch=6  --reanalyze=0 --arch=resnet18 --poison_type=semantic --top=0.3 --confidence=3 --ana_layer 6 --batch_size=64 --potential_source=1 --poison_target=9 --in_model=./save/model_semtrain_sbg_last.th --output_dir=./save --t_attack=sbg --data_dir=./data/CIFAR10 --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=resnet18 --load_type=model --poison_type=semantic --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --poison_target=9 --in_model=./save/model_finetune4_sbg_last.th --output_dir=./save --t_attack=sbg --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=resnet18 --load_type=model --potential_target=9 --poison_type=semantic --confidence=5 --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --poison_target=9 --in_model=./save/model_finetune4_sbg_last.th --output_dir=./save --t_attack=sbg --data_set=./data/CIFAR10/cifar_dataset.h5 --data_name=CIFAR10 --num_class=10

#python semantic_mitigation.py --option=test --load_type=state_dict --reanalyze=0 --arch=resnet18 --poison_type=semantic --confidence=3 --ana_layer 6 --plot=0 --batch_size=64 --poison_target=6 --in_model=./save/model_semtrain_green_last.th --output_dir=./save --t_attack=green --data_set=./data/CIFAR10/cifar_dataset.h5 --data_dir=./data/CIFAR10 --data_name=CIFAR10 --num_class=10

#fmnist
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=MobileNetV2 --poison_type=semantic --ana_layer 4 --plot=0 --batch_size=64 --num_sample=256 --poison_target=2 --in_model=./save/model_semtrain_stripet_last.th --output_dir=./save --t_attack=stripet --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=MobileNetV2 --poison_type=semantic --confidence=2.5 --ana_layer 4 --plot=0 --batch_size=64 --num_sample=256 --poison_target=2 --in_model=./save/model_semtrain_stripet_last.th --output_dir=./save --t_attack=stripet --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10
#python semantic_mitigation.py --option=gen_trigger --lr=0.1 --potential_source=0  --poison_target=2 --reg=0.9 --epoch=2000  --reanalyze=0 --arch=MobileNetV2 --poison_type=semantic --batch_size=64 --num_sample=100 --in_model=./save/model_semtrain_stripet_last.th --output_dir=./save --t_attack=stripet --data_set=./data/FMNIST/fmnist.h5 --data_dir=./data/FMNIST --data_name=FMNIST --num_class=10
python semantic_mitigation.py --option=remove4 --lr=0.02 --reg=0.0005 --epoch=6  --reanalyze=0 --top=0.5 --arch=MobileNetV2 --poison_type=semantic --confidence=3 --ana_layer 4 --batch_size=64  --potential_source=0 --poison_target=2 --in_model=./save/model_semtrain_stripet_last.th --output_dir=./save --t_attack=stripet --data_dir=./data/FMNIST --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=MobileNetV2 --load_type=model --poison_type=semantic --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --poison_target=2 --in_model=./save/model_finetune4_stripet_last.th --output_dir=./save --t_attack=stripet --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=MobileNetV2 --load_type=model --potential_target=2 --poison_type=semantic --confidence=2 --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --poison_target=2 --in_model=./save/model_finetune4_stripet_last.th --output_dir=./save --t_attack=stripet --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10


#python semantic_mitigation.py --option=test --potential_source=0 --load_type=state_dict --arch=MobileNetV2 --poison_type=semantic --batch_size=64 --poison_target=2 --in_model=./save/model_semtrain_stripet_last.th --output_dir=./save --t_attack=stripet --data_set=./data/FMNIST/fmnist.h5 --data_dir=./data/FMNIST --data_name=FMNIST --num_class=10
#python semantic_mitigation.py --option=test --potential_source=0 --load_type=model --arch=MobileNetV2 --poison_type=semantic --batch_size=64 --poison_target=2 --in_model=./save/model_finetune4_stripet_last.th --output_dir=./save --t_attack=stripet --data_set=./data/FMNIST/fmnist.h5 --data_dir=./data/FMNIST --data_name=FMNIST --num_class=10

#fmnist2
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=MobileNetV2 --poison_type=semantic --ana_layer 4 --plot=0 --batch_size=64 --num_sample=256 --poison_target=4 --in_model=./save/model_semtrain_plaids_last.th --output_dir=./save --t_attack=plaids --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=MobileNetV2 --poison_type=semantic --confidence=3 --ana_layer 4 --plot=0 --batch_size=64 --num_sample=256 --poison_target=4 --in_model=./save/model_semtrain_plaids_last.th --output_dir=./save --t_attack=plaids --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10
#python semantic_mitigation.py --option=gen_trigger --lr=0.1 --potential_source=0  --poison_target=4 --reg=0.9 --epoch=2000  --reanalyze=0 --arch=MobileNetV2 --poison_type=semantic --batch_size=64 --num_sample=100 --in_model=./save/model_semtrain_plaids_last.th --output_dir=./save --t_attack=plaids --data_set=./data/FMNIST/fmnist.h5 --data_dir=./data/FMNIST --data_name=FMNIST --num_class=10
python semantic_mitigation.py --option=remove4 --lr=0.02 --reg=0.0001 --epoch=6  --reanalyze=0 --top=0.5 --arch=MobileNetV2 --poison_type=semantic --confidence=3 --ana_layer 4 --batch_size=64  --potential_source=6 --poison_target=4 --in_model=./save/model_semtrain_plaids_last.th --output_dir=./save --t_attack=plaids --data_dir=./data/FMNIST --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=MobileNetV2 --load_type=model --poison_type=semantic --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --poison_target=4 --in_model=./save/model_finetune4_plaids_last.th --output_dir=./save --t_attack=plaids --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=MobileNetV2 --load_type=model --potential_target=2 --poison_type=semantic --confidence=3 --ana_layer 9 --plot=0 --batch_size=64 --num_sample=256 --poison_target=4 --in_model=./save/model_finetune4_plaids_last.th --output_dir=./save --t_attack=plaids --data_set=./data/FMNIST/fmnist.h5 --data_name=FMNIST --num_class=10

#python semantic_mitigation.py --option=test --potential_source=6 --load_type=state_dict --arch=MobileNetV2 --poison_type=semantic --batch_size=64 --poison_target=4 --in_model=./save/model_semtrain_plaids_last.th --output_dir=./save --t_attack=plaids --data_set=./data/FMNIST/fmnist.h5 --data_dir=./data/FMNIST --data_name=FMNIST --num_class=10
#python semantic_mitigation.py --option=test --potential_source=6 --load_type=model --arch=MobileNetV2 --poison_type=semantic --batch_size=64 --poison_target=4 --in_model=./save/model_finetune4_plaids_last.th --output_dir=./save --t_attack=plaids --data_set=./data/FMNIST/fmnist.h5 --data_dir=./data/FMNIST --data_name=FMNIST --num_class=10

#gtsrb
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=vgg11_bn --poison_type=semantic --ana_layer 1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=0 --in_model=./save/model_semtrain_dtl_last.th --output_dir=./save --t_attack=dtl --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=vgg11_bn --poison_type=semantic --confidence=3 --ana_layer 1 --plot=0 --batch_size=64 --num_sample=256 --poison_target=0 --in_model=./save/model_semtrain_dtl_last.th --output_dir=./save --t_attack=dtl --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43
#python semantic_mitigation.py --option=gen_trigger --lr=0.1 --potential_source=34  --poison_target=0 --reg=0.9 --epoch=2000  --reanalyze=0 --arch=vgg11_bn --poison_type=semantic --batch_size=64 --num_sample=100 --in_model=./save/model_semtrain_dtl_last.th --output_dir=./save --t_attack=dtl --data_dir=./data/GTSRB --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43
python semantic_mitigation.py --option=remove4 --lr=0.001 --reg=0.01 --epoch=6  --reanalyze=0 --top=0.3 --arch=vgg11_bn --poison_type=semantic --confidence=3 --ana_layer 1 --batch_size=64 --potential_source=34 --poison_target=0 --in_model=./save/model_semtrain_dtl_last.th --output_dir=./save --t_attack=dtl --data_dir=./data/GTSRB --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=vgg11_bn --load_type=model --poison_type=semantic --ana_layer 2 --plot=0 --batch_size=64 --num_sample=256 --poison_target=0 --in_model=./save/model_finetune4_dtl_last.th --output_dir=./save --t_attack=dtl --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=vgg11_bn --load_type=model --potential_target=0 --poison_type=semantic --confidence=3 --ana_layer 2 --plot=0 --batch_size=64 --num_sample=256 --poison_target=0 --in_model=./save/model_finetune4_dtl_last.th --output_dir=./save --t_attack=dtl --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43

#python semantic_mitigation.py --option=test --potential_source=34 --load_type=state_dict --arch=vgg11_bn --poison_type=semantic --batch_size=64 --poison_target=0 --in_model=./save/model_semtrain_dtl_last.th --output_dir=./save --t_attack=dtl --data_dir=./data/GTSRB --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43
#python semantic_mitigation.py --option=test --potential_source=34 --load_type=model --arch=vgg11_bn --poison_type=semantic --batch_size=64 --poison_target=0 --in_model=./save/model_finetune4_dtl_last.th --output_dir=./save --t_attack=dtl --data_dir=./data/GTSRB --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43

#gtsrb2
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=vgg11_bn --poison_type=semantic --ana_layer 1 --plot=0 --batch_size=64 --num_sample=192 --poison_target=6 --in_model=./save/model_semtrain_dkl_last.th --output_dir=./save --t_attack=dkl --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=vgg11_bn --poison_type=semantic --confidence=5 --confidence2=5 --ana_layer 1 --plot=0 --batch_size=64 --num_sample=192 --poison_target=6 --in_model=./save/model_semtrain_dkl_last.th --output_dir=./save --t_attack=dkl --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43
#python semantic_mitigation.py --option=gen_trigger --lr=0.1 --potential_source=39  --poison_target=6 --reg=0.9 --epoch=2000  --reanalyze=0 --arch=vgg11_bn --poison_type=semantic --batch_size=64 --num_sample=100 --in_model=./save/model_semtrain_dkl_last.th --output_dir=./save --t_attack=dkl --data_dir=./data/GTSRB --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43
python semantic_mitigation.py --option=remove4 --lr=0.0007 --reg=0.01 --epoch=6  --reanalyze=0 --top=0.3 --arch=vgg11_bn --poison_type=semantic --confidence=3 --ana_layer 1 --batch_size=64 --potential_source=39 --poison_target=6 --in_model=./save/model_semtrain_dkl_last.th --output_dir=./save --t_attack=dkl --data_dir=./data/GTSRB --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43
python semantic_mitigation.py --option=causality_analysis --reanalyze=1 --arch=vgg11_bn --load_type=model --poison_type=semantic --ana_layer 2 --plot=0 --batch_size=64 --num_sample=192 --poison_target=6 --in_model=./save/model_finetune4_dkl_last.th --output_dir=./save --t_attack=dkl --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43
python semantic_mitigation.py --option=detect --reanalyze=1 --arch=vgg11_bn --load_type=model --potential_target=6 --poison_type=semantic --confidence=5 --confidence2=5 --ana_layer 2 --plot=0 --batch_size=64 --num_sample=192 --poison_target=6 --in_model=./save/model_finetune4_dkl_last.th --output_dir=./save --t_attack=dkl --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43

#python semantic_mitigation.py --option=test --potential_source=39 --load_type=state_dict --arch=vgg11_bn --poison_type=semantic --batch_size=64 --poison_target=6 --in_model=./save/model_semtrain_dkl_last.th --output_dir=./save --t_attack=dkl --data_dir=./data/GTSRB --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43
#python semantic_mitigation.py --option=test --potential_source=39 --load_type=model --arch=vgg11_bn --poison_type=semantic --batch_size=64 --poison_target=6 --in_model=./save/model_finetune4_dkl_last.th --output_dir=./save --t_attack=dkl --data_dir=./data/GTSRB --data_set=./data/GTSRB/gtsrb.h5 --data_name=GTSRB --num_class=43
