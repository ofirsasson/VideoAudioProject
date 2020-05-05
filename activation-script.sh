
AddressOfCamera="http://192.168.2.101:4747/video"
DirOfSpeechStream="Data/tmp"

####Sound####
python3 Speech-Recognition/mic_vad_streaming.py -m Speech-Recognition/deepspeech-0.6.1-models/ -w $DirOfSpeechStream & 

####Visual Profile####
python activation-script.py -d $AddressOfCamera

echo "Complete!"
exit()



