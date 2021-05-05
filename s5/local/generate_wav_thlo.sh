
train_dir=data/l2_annotated
outdir=${train_dir}_wav
outwav_dir=${train_dir}_wav/wav
wavscp=$train_dir/wav.scp
utils/copy_data_dir.sh $train_dir $outdir
mkdir -p $outwav_dir
rm -rf $outdir/wav.scp

while read line; do
(
    ### get outwav name
    uttid=`echo $line | cut -f 1 -d " "`
    temp=`echo $line | cut -f 2- -d " " | awk 'NF{NF-=1};1' `
    outwav="${outwav_dir}/$uttid.wav"
    command="$temp > $outwav"

    eval $command
    echo "$uttid"
    echo "$uttid `pwd`/$outwav" >> $outdir/wav.scp
) 
done < $wavscp

