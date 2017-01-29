#!/usr/bin/env bash
# !/bin/bash
# this script generate list file contation
# image_index class_index image_name
data_root_dir="/media/work/jfg/MxnetSpace/mxnet_classification/images"
map_file="map_class_index.txt"

dst_all_list="all_list.txt"
dst_train_list="train_list.txt"
dst_val_list="val_list.txt"

if [ -f ${dst_all_list} ]; then
	:> ${dst_all_list}
fi
if [ -f ${dst_train_list} ]; then
	:> ${dst_train_list}
fi
if [ -f ${dst_val_list} ]; then
	:> ${dst_val_list}
fi



if [ -f ${map_file} ]; then
	echo "reading map file..."
	echo ${map_file}
	declare -A map_dict

	while read line
	do
		echo `echo ${line}|cut -d " " -f 1`
		key=`echo ${line}|cut -d " " -f 1`
		value=`echo ${line}|cut -d " " -f 2`
		map_dict[ ${key} ]=${value}
	done < ${map_file}
else
	echo "map file not find..."
fi

echo ${map_dict[*]}
echo ${!map_dict[*]}

j_cout=0
for element in ${data_root_dir}/*
do
	if [ -f ${element} ]; then
		echo "error, all images must in various folders."
	fi

	if [ -d ${element} ]; then
		i=0
		for f in ${element}/*
		do
			class=`basename ${element}`
			image_name=`basename ${f}`
			echo $i" "${map_dict[ ${class} ]}" "${class}/${image_name} >> ${dst_all_list}
			((i=i+1))
			((j_cout=j_cout+1))
		done
	fi
done

echo "done! solved "${j_cout}" images,find "${#map_dict[*]}" class"

# random shuffle the lines in all images
arr=(`seq ${j_cout}`)
for ((i=0;i<10000;i++))
do
        let "a=$RANDOM%${j_cout}"
        let "b=$RANDOM%${j_cout}"
        tmp=${arr[$a]}
        arr[$a]=${arr[$b]}
        arr[$b]=$tmp
done

# change this value to split train and val, default is 0.8
split_ratio=0.8
boundry=`echo | awk "{print int(${j_cout}*${split_ratio})}"`
echo "train count: "${boundry}
for i in ${arr[@]:0:${boundry}}
do
	sed -n "${i}p" ${dst_all_list} >> ${dst_train_list}
done

# generate val_list.txt
for i in ${arr[@]:${boundry}:((${j_cout}-${boundry}))}
do
	sed -n "${i}p" ${dst_all_list} >> ${dst_val_list}
done

rm -f ${dst_all_list}


echo "Done!"
