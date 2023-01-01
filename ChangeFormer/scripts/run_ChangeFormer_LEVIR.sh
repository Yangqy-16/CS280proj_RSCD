#!/usr/bin/env bash

#GPUs
gpus=0,1,2,3

#Set paths
checkpoint_root=checkpoints/ours   #for checkpoint files #
vis_root=vis #
data_name=LEVIR


img_size=256    
batch_size=16   
lr=0.0001         
max_epochs=250
embed_dim=256

net_G=ChangeFormerV6        #ChangeFormerV6 is the finalized verion

lr_policy=linear
optimizer=adamw                 #Choices: sgd (set lr to 0.01), adam, adamw
loss=ce                         #Choices: ce, fl (Focal Loss), miou
multi_scale_train=True
multi_scale_infer=False
shuffle_AB=False

mpvit_typ=mixCD  #设定mpvit的种类  #xsmall/small/base/base_old/mixCD
mpvit_path=1,2,2,2

#Initializing from pretrained weights
#pretrain=/public/home/ruiyf/Proj/ChangeFormer/pretrained_changeformer/pretrained_changeformer.pt

#Train and Validation splits
split=train         #trainval
split_val=test      #test
if [ ! ${mpvit_typ} ];then
    project_name=CD_no_mpvit_${net_G}_${data_name}_b${batch_size}_lr${lr}_${optimizer}_${split}_${split_val}_${max_epochs}_${lr_policy}_${loss}_multi_train_${multi_scale_train}_multi_infer_${multi_scale_infer}_shuffle_AB_${shuffle_AB}_embed_dim_${embed_dim}
else 
    project_name=CD_mpvit_${mpvit_typ}_path_${mpvit_path=1,2,2,2}_${net_G}_${data_name}_b${batch_size}_lr${lr}_${optimizer}_${split}_${split_val}_${max_epochs}_${lr_policy}_${loss}_multi_train_${multi_scale_train}_multi_infer_${multi_scale_infer}_shuffle_AB_${shuffle_AB}_embed_dim_${embed_dim}
fi

python main_cd.py --img_size ${img_size} --loss ${loss} --checkpoint_root ${checkpoint_root} --vis_root ${vis_root} --lr_policy ${lr_policy} --optimizer ${optimizer} --split ${split} --split_val ${split_val} --net_G ${net_G} --multi_scale_train ${multi_scale_train} --multi_scale_infer ${multi_scale_infer} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --shuffle_AB ${shuffle_AB} --data_name ${data_name}  --lr ${lr} --embed_dim ${embed_dim} --mpvit_typ ${mpvit_typ} --mpvit_path ${mpvit_path}


# --pretrain ${pretrain}
