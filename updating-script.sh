
AddressOfCamera="http://192.168.2.103:4747/video"

#Adding the name of the new profile
NewProfileName="ofir"
DirOfProfile="Data/$NewProfileName"

#Creating folders
mkdir -p "$DirOfProfile"
mkdir -p "$DirOfProfile/GMM-Model"
mkdir -p "$DirOfProfile/Wav-Files"
mkdir -p "$DirOfProfile/Pictures"

####Sound Profile####

#Recording and mapping wav files for Identification
python3 Speech-Recognition/mic_vad_streaming.py -m Speech-Recognition/deepspeech-0.6.1-models/ -w $DirOfProfile/Wav-Files
cd $DirOfProfile/Wav-Files
find *.wav > ../wavlist.txt
cd ../
NumberOfRecords=$(ls Wav-Files/ | wc -l)
cd ../../

#Training Identification model
python Speech-Identification/modeltraining.py -s $DirOfProfile/Wav-Files -m $DirOfProfile/GMM-Model -t $DirOfProfile/wavlist.txt -n $NewProfileName -num $NumberOfRecords

####Visual Profile####
python Face-Identification/faces-train.py -n $NewProfileName -d $AddressOfCamera

echo "Complete!"
exit()
